"""
Influence-Encoder Training (Gradients Only)
==========================================
Trains an encoder on pre-stocked gradients and evaluates on:
  - Track 1: Native gradient similarity (Spearman correlation)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
import numpy as np
import argparse
import warnings
import logging
import math
import json

# Make influence_eval importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import time
from influence_eval.flops_measure import flop_counter, save_phase_flops, save_phase_timing
from typing import List, Dict, Sequence
from collections import deque
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import spearmanr
from transformers import set_seed, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models

# Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# =========================================================================
# Run Modes Configuration
# =========================================================================
MODES = {
    'tiny':   {'train_a': 3,    'eval_a': 2,    'train_p': 5,    'eval_p': 3,    'epochs': 1, 'max_train_pool': 10},
    'quick':  {'train_a': 100,  'eval_a': 100,  'train_p': 200,  'eval_p': 200,  'epochs': 2, 'max_train_pool': 5000},
    'small':  {'train_a': 2000,  'eval_a': 50,  'train_p': 4000, 'eval_p': 50,  'epochs': 10, 'max_train_pool': 10000},
    'medium': {'train_a': 2000, 'eval_a': 500,  'train_p': 6000, 'eval_p': 1000, 'epochs': 5, 'max_train_pool': 10000},
    'full':   {'train_a': 4000, 'eval_a': 1000, 'train_p': 16000, 'eval_p': 4000, 'epochs': 2, 'max_train_pool': None}
}

# =========================================================================
# Formatting & Diagnostic Utilities
# =========================================================================

class RunningAverage:
    def __init__(self, alpha=0.1, window_size=100):
        self.alpha = alpha
        self.ema_value = None
        self.window = deque(maxlen=window_size)
    
    def update(self, val):
        if self.ema_value is None: self.ema_value = val
        else: self.ema_value = self.alpha * val + (1 - self.alpha) * self.ema_value
        self.window.append(val)
    
    def ema(self):
        return self.ema_value if self.ema_value is not None else 0.0

def check_gradient_diagnostics(name: str, grads: np.ndarray):
    if grads.size == 0:
        raise ValueError(f"[Diag] {name}: EMPTY ARRAY")
    
    nans = np.isnan(grads).sum()
    infs = np.isinf(grads).sum()
    if nans > 0 or infs > 0:
        raise ValueError(f"[Diag] {name}: Found {nans} NaNs and {infs} Infs – aborting")
    
    zeros = np.sum(grads == 0.0)
    print(f"[Diag] {name}: shape={grads.shape}, dtype={grads.dtype}")
    print(f"          -> NaNs: {nans} | Infs: {infs} | Exact Zeros: {zeros}")
    
    valid_grads = grads[np.isfinite(grads)]
    if valid_grads.size > 0:
        print(f"          -> Mean: {valid_grads.mean():.6e} | Std: {valid_grads.std():.6e}")

# =========================================================================
# Stocked-split Loading (file-based; output of gradient_stocking_EXACT.py)
# =========================================================================

def load_stocked_split(prefix: str) -> tuple[list[dict], np.ndarray, dict]:
    """Load a stocked split written by gradient_stocking_EXACT.py.

    Expects three files at the given prefix:
      {prefix}_grads.pt    — torch tensor [N, proj_dim], L2-normalized
      {prefix}_inputs.json — list of N input dicts (same order as grads)
      {prefix}_meta.json   — metadata (proj_dim, formatter, split, etc.)

    Returns (input_dicts, grads_np, meta).  The input dicts are exactly the
    structures passed to construct_test_sample (anchors) or
    encode_with_messages_format (pool) during stocking — so they can be
    re-tokenized identically here for the GT diagnostic.
    """
    grads_path = f"{prefix}_grads.pt"
    inputs_path = f"{prefix}_inputs.json"
    meta_path = f"{prefix}_meta.json"
    for pth in (grads_path, inputs_path, meta_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(
                f"❌ Stocked split file not found: {pth}\n"
                f"   Run gradient_stocking_EXACT.py for this split first."
            )
    grads = torch.load(grads_path, weights_only=False)
    if isinstance(grads, torch.Tensor):
        grads_np = grads.cpu().float().numpy()
    else:
        grads_np = np.asarray(grads, dtype=np.float32)
    with open(inputs_path, "r", encoding="utf-8") as f:
        input_dicts = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if len(input_dicts) != grads_np.shape[0]:
        raise ValueError(
            f"Length mismatch in {prefix}: {len(input_dicts)} inputs vs {grads_np.shape[0]} grads"
        )
    return input_dicts, grads_np, meta


_ENCODER_PREFIX = (
    "Instruct: Given a sample, find the passages closest to that sample.\nQuery:"
)


def anchor_dicts_to_texts(input_dicts: list[dict]) -> list[str]:
    """Format BBH anchor input dicts {"prompts": p, "labels": r} for the encoder.

    Matches bbh_texts_for_encoder() in bbh_data.py so the encoder sees identical
    text at training time and final-pipeline-eval time.
    """
    return [f"{_ENCODER_PREFIX} {d['prompts']} {d['labels']}".strip() for d in input_dicts]


def pool_dicts_to_texts(input_dicts: list[dict], tokenizer=None) -> list[str]:
    """Format Tulu pool input dicts {"messages": [...]} for the encoder.

    Renders via the same logic as _concat_messages() in compute_sentence_embeds.py
    so encoder text matches the final-pipeline-eval format exactly.
    """
    eos = tokenizer.eos_token if (tokenizer and tokenizer.eos_token) else ""
    texts: list[str] = []
    for d in input_dicts:
        msgs = d.get("messages", [])
        parts: list[str] = []
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "").strip()
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}{eos}\n")
        texts.append("".join(parts).strip())
    return texts

# =========================================================================
# Native Gradient Metrics Logic
# =========================================================================

def aggregate_scores(score_matrix, mode='mean'):
    if score_matrix.size == 0: return np.array([])
    if mode == 'mean': return score_matrix.mean(axis=0)
    elif mode == 'max': return score_matrix.max(axis=0)
    raise ValueError(f"Unknown mode: {mode}")

def safe_normalize(tensor, dim=1, eps=1e-8):
    if tensor.ndim == 1:
        # Handle 1D tensor (single sample or empty) by adding a batch dimension
        tensor = tensor.unsqueeze(0)
    
    if tensor.size(0) == 0:
        return tensor

    tensor_d = tensor.to(torch.float64)
    norms = torch.norm(tensor_d, p=2, dim=dim, keepdim=True)
    norms = torch.clamp(norms, min=eps)
    normalized = (tensor_d / norms).to(tensor.dtype)
    return normalized

def compute_native_metrics(true_scores, method_scores, agg_mode='mean'):
    if true_scores.size == 0 or method_scores.size == 0:
        return {'agg_spearman': 0.0, 'per_anchor_spearman_mean': 0.0}
    
    true_scores = np.nan_to_num(true_scores, nan=0.0)
    method_scores = np.nan_to_num(method_scores, nan=0.0)
    
    true_agg = aggregate_scores(true_scores, mode=agg_mode)
    method_agg = aggregate_scores(method_scores, mode=agg_mode)
    
    if true_agg.size < 2 or np.std(true_agg) < 1e-9 or np.std(method_agg) < 1e-9:
        agg_spearman = 0.0
    else:
        agg_spearman, _ = spearmanr(true_agg, method_agg)
    
    per_anchor_spearman = []
    n_anchors = true_scores.shape[0] if len(true_scores.shape) > 1 else 1
    for i in range(n_anchors):
        if np.std(true_scores[i]) < 1e-9 or np.std(method_scores[i]) < 1e-9: continue
        corr, _ = spearmanr(true_scores[i], method_scores[i])
        if not np.isnan(corr): per_anchor_spearman.append(corr)
    
    return {
        'agg_spearman': float(agg_spearman),
        'per_anchor_spearman_mean': float(np.mean(per_anchor_spearman)) if per_anchor_spearman else 0.0,
    }

def quick_eval_native(encoder, eval_anchors_text, eval_text, true_scores_eval, agg_mode, loss_fn, device, batch_size=8, n_candidates=16):
    if not eval_anchors_text or not eval_text:
        return {'agg_spearman': 0.0, 'per_anchor_mean': 0.0, 'loss': 0.0}
    encoder.eval()
    with torch.inference_mode():
        z_a = encoder.encode(eval_anchors_text, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, device=device)
        z_e = encoder.encode(eval_text, convert_to_tensor=True, normalize_embeddings=True, batch_size=32, device=device)
        
        # Compute "global" scores for Spearman metrics
        full_scores = torch.mm(z_a, z_e.t())
        full_labels = torch.from_numpy(true_scores_eval).to(device)
        
        # Compute "batch-comparable" loss by sampling blocks of the same size as training
        # This ensures Pearson/MSE scales match what the training EMA sees.
        eval_losses = []
        n_eval_batches = 10 # Sample 10 batches for a stable estimate
        
        n_a, n_p = z_a.shape[0], z_e.shape[0]
        for _ in range(n_eval_batches):
            # Randomly sample indices for a "virtual batch"
            idx_a = torch.randperm(n_a)[:batch_size]
            idx_p = torch.randperm(n_p)[:n_candidates]
            
            batch_scores = full_scores[idx_a][:, idx_p]
            batch_labels = full_labels[idx_a][:, idx_p]
            
            eval_losses.append(loss_fn(batch_scores, batch_labels).item())
            
        eval_loss = np.mean(eval_losses)
        scores_np = full_scores.cpu().numpy()
        
    metrics = compute_native_metrics(true_scores_eval, scores_np, agg_mode=agg_mode)
    encoder.train()
    return {
        'agg_spearman': metrics['agg_spearman'], 
        'per_anchor_mean': metrics['per_anchor_spearman_mean'],
        'loss': eval_loss
    }

# =========================================================================
# Training Components
# =========================================================================

class InBatchLoss(nn.Module):
    """Hybrid loss: Pearson correlation + [KL-Divergence or MSE].

    Optimising Pearson r helps global alignment. 
    - 'kl' mode (default) adds listwise ranking via KL-Divergence.
    - 'mse' mode adds scale alignment via MSE (ablation).
    """

    def __init__(self, alpha=0.5, temperature=0.05, mode='kl'):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mode = mode

    def forward(self, scores, labels):
        # 1. Pearson Correlation Loss
        s_flat = scores.view(-1)
        l_flat = labels.view(-1)
        s_cent = s_flat - s_flat.mean()
        l_cent = l_flat - l_flat.mean()
        cov = (s_cent * l_cent).sum()
        s_std = torch.sqrt((s_cent ** 2).sum() + 1e-8)
        l_std = torch.sqrt((l_cent ** 2).sum() + 1e-8)
        pearson_loss = 1.0 - (cov / (s_std * l_std))

        if self.mode == 'kl':
            # 2a. KL-Divergence Loss (Listwise continuous ranking)
            pred_logp = F.log_softmax(scores / self.temperature, dim=1)
            true_p = F.softmax(labels / self.temperature, dim=1)
            second_loss = F.kl_div(pred_logp, true_p, reduction='batchmean')
        else:
            # 2b. MSE Loss (Ablation)
            # Standardize labels to match score scale for MSE stability
            l_standardized = l_cent / (l_std / s_std.detach() + 1e-8) + s_flat.mean().detach()
            second_loss = F.mse_loss(s_flat, l_standardized)

        return self.alpha * pearson_loss + (1 - self.alpha) * second_loss

class AnchorBatchGenerator:
    def __init__(self, anchor_texts, pool_texts, true_scores, k_anchors=8, m_candidates=64, hard_ratio=0.5, seed=42):
        self.anchor_texts = anchor_texts
        self.pool_texts = pool_texts
        self.true_scores = true_scores
        self.k = k_anchors
        self.m = m_candidates
        self.hard_ratio = hard_ratio
        self.rng = random.Random(seed)
        self.n_anchors = len(anchor_texts)
        self.n_pool = len(pool_texts)
        self.anchor_order = list(range(self.n_anchors))
        self.rng.shuffle(self.anchor_order)
        self.sorted_indices = np.argsort(-self.true_scores, axis=1) if self.true_scores.size > 0 else None
    
    def _sample_candidates(self, anchor_indices):
        if self.sorted_indices is None:
            return self.rng.sample(range(self.n_pool), min(self.m, self.n_pool))
        n_hard = int(self.m * self.hard_ratio)
        hard_candidates = set()
        top_per_anchor = max(1, n_hard // len(anchor_indices))
        for idx in anchor_indices:
            hard_candidates.update(self.sorted_indices[idx, :top_per_anchor * 2].tolist())
        hard_candidates = list(hard_candidates)
        if len(hard_candidates) > n_hard: hard_candidates = self.rng.sample(hard_candidates, n_hard)
        remaining =[i for i in range(self.n_pool) if i not in hard_candidates]
        random_candidates = self.rng.sample(remaining, min(self.m - n_hard, len(remaining)))
        all_candidates = hard_candidates + random_candidates
        self.rng.shuffle(all_candidates)
        return all_candidates[:self.m]
    
    def __iter__(self):
        for batch_start in range(0, self.n_anchors, self.k):
            batch_end = min(batch_start + self.k, self.n_anchors)
            anchor_indices = self.anchor_order[batch_start:batch_end]
            candidate_indices = self._sample_candidates(anchor_indices)
            labels = self.true_scores[np.ix_(anchor_indices, candidate_indices)]
            yield ([self.anchor_texts[i] for i in anchor_indices], 
                   [self.pool_texts[i] for i in candidate_indices], 
                   torch.tensor(labels, dtype=torch.float32))
    
    def __len__(self):
        return (self.n_anchors + self.k - 1) // self.k

# =========================================================================
# Save Encoder with Metadata
# =========================================================================

def save_encoder_with_metadata(encoder, output_dir, method_name="influence_encoder", args=None, metrics=None):
    """Save encoder model with standardized structure and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the encoder model
    model_dir = os.path.join(output_dir, "model")
    encoder.save(model_dir)
    
    # Save metadata
    metadata = {
        "method": method_name,
        "encoder_base": args.encoder_model if args else "unknown",
        "gradient_seed": args.gradient_seed if args else 42,
        "run_mode": args.run_mode if args else "unknown",
        "timestamp": str(torch.cuda.Event(enable_timing=False)) if torch.cuda.is_available() else __import__('datetime').datetime.now().isoformat(),
        "metrics": metrics if metrics else {}
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Encoder saved to {output_dir}")
    print(f"   - Model: {model_dir}")
    print(f"   - Metadata: metadata.json")

# =========================================================================
# Main Execution
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default="quick", choices=['tiny', 'quick', 'small', 'medium', 'full'])
    parser.add_argument('--encoder_model', type=str, default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument('--gradient_model', type=str, default="Qwen/Qwen3-0.6B-Base",
                        help="Model whose chat template is used to format text (must match gradient_stocking).")
    parser.add_argument('--anchor_train_prefix', type=str, required=True,
                        help="Prefix (no extension) of the stocked train_anchors files written by gradient_stocking_EXACT.py.")
    parser.add_argument('--anchor_eval_prefix', type=str, required=True)
    parser.add_argument('--pool_train_prefix', type=str, required=True)
    parser.add_argument('--pool_eval_prefix', type=str, required=True)
    parser.add_argument('--gradient_seed', type=int, default=42,
                        help="Kept for backward-compat naming; the projection seed is set at stocking time.")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--n_candidates_per_batch', type=int, default=15)
    parser.add_argument('--hard_ratio', type=float, default=0.0)
    parser.add_argument('--agg_mode', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for Pearson loss (1-alpha for secondary loss)")
    parser.add_argument('--loss_mode', type=str, default='kl', choices=['kl', 'mse'], help="Secondary loss: 'kl' (ranking) or 'mse' (ablation)")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/influence_encoder")
    # LoRA / GT config — must match gradient_stocking_EXACT so the end-of-training
    # GT diagnostic computes gradients with the same fresh-LoRA as the stocking did.
    parser.add_argument('--lora_rank', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=512)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_seed', type=int, default=0)
    parser.add_argument('--lora_target_modules', type=str, default='all-linear')
    parser.add_argument('--gt_proj_dim', type=int, default=65536,
                        help='Projection dim for end-of-training GT scores (typically GT_PROJ_DIM from config).')
    parser.add_argument('--project_interval', type=int, default=1)
    parser.add_argument('--eval_gt', action='store_true',
                        help='Run the expensive LESS-style ground-truth gradient evaluation at the end.'
                             ' Off by default; enable when you need the GT columns.')


    args = parser.parse_args()
    cfg = MODES[args.run_mode]
    epochs = args.epochs if args.epochs is not None else cfg['epochs']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(137)
    
    print("=" * 70)
    print(f"🔬 Influence-Encoder Training (Mode: {args.run_mode.upper()})")
    print("=" * 70)
    
    # 1. Load Data (stocked splits = .pt + .json + meta.json triplets)
    print(f"\n🔤 Loading gradient tokenizer for text formatting: {args.gradient_model}")
    gradient_tokenizer = AutoTokenizer.from_pretrained(args.gradient_model)
    if getattr(gradient_tokenizer, "chat_template", None) in (None, ""):
        gradient_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
    if getattr(gradient_tokenizer, "pad_token", None) is None:
        gradient_tokenizer.pad_token = gradient_tokenizer.eos_token

    print("\n📥 Loading stocked splits...")
    train_anchors, grads_train_anchors, meta_ta = load_stocked_split(args.anchor_train_prefix)
    eval_anchors,  grads_eval_anchors,  meta_ea = load_stocked_split(args.anchor_eval_prefix)
    train_pool,    grads_train_pool,    meta_tp = load_stocked_split(args.pool_train_prefix)
    eval_pool,     grads_eval_pool,     meta_ep = load_stocked_split(args.pool_eval_prefix)

    print(f"   📊 Loaded sizes — Train Anchors: {len(train_anchors)}, Eval Anchors: {len(eval_anchors)}, "
          f"Train Pool: {len(train_pool)}, Eval Pool: {len(eval_pool)}")
    print(f"   📊 Formatter — anchors: {meta_ta.get('formatter')}, pool: {meta_tp.get('formatter')}")
    print(f"   📊 CountSketch proj_dim = {meta_ta.get('proj_dim')}")

    # Sub-sample to the configured mode sizes
    def _take(items, grads, n):
        n = min(n, len(items))
        return items[:n], grads[:n]

    train_anchors, grads_train_anchors = _take(train_anchors, grads_train_anchors, cfg['train_a'])
    eval_anchors,  grads_eval_anchors  = _take(eval_anchors,  grads_eval_anchors,  cfg['eval_a'])
    train_pool,    grads_train_pool    = _take(train_pool,    grads_train_pool,    cfg['train_p'])
    eval_pool,     grads_eval_pool     = _take(eval_pool,     grads_eval_pool,     cfg['eval_p'])

    check_gradient_diagnostics("Train Anchors", grads_train_anchors)
    check_gradient_diagnostics("Eval Anchors",  grads_eval_anchors)
    check_gradient_diagnostics("Train Pool",    grads_train_pool)
    check_gradient_diagnostics("Eval Pool",     grads_eval_pool)

    train_anchors_text = anchor_dicts_to_texts(train_anchors)
    eval_anchors_text  = anchor_dicts_to_texts(eval_anchors)
    train_text         = pool_dicts_to_texts(train_pool, gradient_tokenizer)
    eval_text          = pool_dicts_to_texts(eval_pool,  gradient_tokenizer)
    
    # Pre-calculate Ground Truth Scores (single CountSketch seed for both train and eval)
    true_scores_train = torch.mm(
        safe_normalize(torch.from_numpy(grads_train_anchors)),
        safe_normalize(torch.from_numpy(grads_train_pool)).t()
    ).numpy()
    true_scores_eval = torch.mm(
        safe_normalize(torch.from_numpy(grads_eval_anchors)),
        safe_normalize(torch.from_numpy(grads_eval_pool)).t()
    ).numpy()

    # 2. Training Loop
    enc = SentenceTransformer(args.encoder_model, device=device)
    enc.max_seq_length = 1024  # data is filtered to 512 Qwen tokens; 1024 is a safe ceiling for the encoder's own tokenizer
    enc.train()
    loss_fn = InBatchLoss(alpha=args.alpha, mode=args.loss_mode)
    optimizer = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    n_batches = (len(train_anchors_text) + args.batch_size - 1) // args.batch_size
    total_steps = math.ceil(n_batches * epochs / args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    train_loss_avg = RunningAverage()
    global_batch_idx = 0

    training_total_flops = 0
    training_total_time_s = 0.0
    for epoch in range(epochs):
        gen = AnchorBatchGenerator(train_anchors_text, train_text, true_scores_train,
                                   args.batch_size, args.n_candidates_per_batch,
                                   args.hard_ratio, seed=42+epoch)
        pbar = tqdm(gen, desc=f"Epoch {epoch+1}/{epochs}", total=n_batches)

        t0_epoch = time.perf_counter()
        with flop_counter() as epoch_counter:
            for batch_idx, (a_text, c_text, labels) in enumerate(pbar):
                tokenized_a = enc.tokenizer(a_text, padding=True, truncation=True, max_length=enc.max_seq_length, return_tensors="pt").to(device)
                tokenized_c = enc.tokenizer(c_text, padding=True, truncation=True, max_length=enc.max_seq_length, return_tensors="pt").to(device)

                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    a_embs = F.normalize(enc(tokenized_a)['sentence_embedding'], p=2, dim=1)
                    c_embs = F.normalize(enc(tokenized_c)['sentence_embedding'], p=2, dim=1)
                    loss = loss_fn(torch.mm(a_embs, c_embs.T), labels.to(device)) / args.grad_accum_steps

                scaler.scale(loss).backward()
                train_loss_avg.update(loss.item() * args.grad_accum_steps)
                global_batch_idx += 1

                # Free intermediate tensors immediately to prevent memory accumulation
                del tokenized_a, tokenized_c, a_embs, c_embs, loss

                if global_batch_idx % args.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Periodically flush the CUDA allocator cache
                if global_batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

                pbar.set_postfix(ema_loss=f"{train_loss_avg.ema():.4f}")

            if global_batch_idx % args.grad_accum_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        training_total_flops += int(epoch_counter.get_total_flops())
        training_total_time_s += time.perf_counter() - t0_epoch

        # Quick Epoch Eval (not counted toward training FLOPs)
        ev = quick_eval_native(enc, eval_anchors_text[:50], eval_text[:200], true_scores_eval[:50, :200],
                               args.agg_mode, loss_fn, device,
                               batch_size=args.batch_size, n_candidates=args.n_candidates_per_batch)
        print(f"\n   [Epoch {epoch+1}] Loss - Train: {train_loss_avg.ema():.4f} | Eval: {ev['loss']:.4f}")
        print(f"   [Epoch {epoch+1}] Eval Spearman - Agg ρ: {ev['agg_spearman']:.4f} | PA ρ: {ev['per_anchor_mean']:.4f}")

        # Release all epoch-level memory before the next epoch
        gc.collect()
        torch.cuda.empty_cache()

    training_flops = training_total_flops
    training_time_s = training_total_time_s

    # 3. Final Evaluation (not counted toward training FLOPs)
    print("\n" + "=" * 70 + "\n🏁 FINAL EVALUATION\n" + "=" * 70)
    enc.eval()
    base_enc = SentenceTransformer(args.encoder_model, device=device)
    base_enc.max_seq_length = enc.max_seq_length

    with torch.inference_mode():
        enc_a  = enc.encode(eval_anchors_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        enc_e  = enc.encode(eval_text,         normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        base_a = base_enc.encode(eval_anchors_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        base_e = base_enc.encode(eval_text,         normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
    enc_pred  = enc_a  @ enc_e.T
    base_pred = base_a @ base_e.T

    # Offload encoders before loading gradient model for GT computation
    enc.to("cpu")
    base_enc.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    sketch_dim = grads_eval_anchors.shape[1]
    enc_sketch  = compute_native_metrics(true_scores_eval, enc_pred,  agg_mode=args.agg_mode)
    base_sketch = compute_native_metrics(true_scores_eval, base_pred, agg_mode=args.agg_mode)

    flops_path = os.path.join(args.output_dir, "_flops.json")
    save_phase_flops(flops_path, training_flops)
    timing_path = os.path.join(args.output_dir, "_timing.json")
    save_phase_timing(timing_path, training_time_s)
    print(f"   📊 Training FLOPs: {training_flops:.3e}  ({flops_path})")
    print(f"   ⏱  Training time:  {training_time_s:.1f}s  ({timing_path})")
    print(f"   CountSketch dim={sketch_dim} | eval anchors={len(eval_anchors_text)} pool={len(eval_text)}")

    # --- Diagnostic: compute LESS-style GT scores on the same eval samples.
    # Input dicts (eval_anchors, eval_pool) were stored at stocking time with the
    # exact shape that construct_test_sample / encode_with_messages_format expect,
    # so we pass them directly — guaranteed identical tokenization vs the stocking.
    print("\n🔬 Computing LESS-style GT scores for eval samples (same points, high proj_dim)...")
    from datasets import Dataset as HFDataset
    from common.data import construct_test_sample, encode_with_messages_format
    from influence_eval.model_utils import load_base_with_fresh_lora
    from representation.less.compute_less_embeds import collect_grads, normalize_embeddings_in_chunks
    from representation.helper import batch_cosine_similarity

    anchor_hf = HFDataset.from_list(eval_anchors)
    anchor_hf = anchor_hf.map(
        lambda x: construct_test_sample(tokenizer=gradient_tokenizer, sample=x, max_length=2048),
        num_proc=1, load_from_cache_file=False,
    )
    anchor_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    pool_hf = HFDataset.from_list(eval_pool)
    pool_hf = pool_hf.map(
        lambda x: encode_with_messages_format(x, gradient_tokenizer, max_seq_length=2048, include_response=True),
        num_proc=1, load_from_cache_file=False,
    )
    pool_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    grad_model = load_base_with_fresh_lora(
        model_name=args.gradient_model,
        tokenizer=gradient_tokenizer,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.lora_seed,
    )
    grad_model.eval()

    anchor_dl = torch.utils.data.DataLoader(anchor_hf, batch_size=1, shuffle=False)
    pool_dl   = torch.utils.data.DataLoader(pool_hf,   batch_size=1, shuffle=False)

    print(f"   Anchor gradients ({len(eval_anchors)} samples, proj_dim={args.gt_proj_dim})...")
    a_grads = collect_grads(anchor_dl, grad_model, proj_dim=args.gt_proj_dim,
                            adam_optimizer_state=None, gradient_type="sgd",
                            project_interval=args.project_interval)
    print(f"   Pool gradients ({len(eval_pool)} samples, proj_dim={args.gt_proj_dim})...")
    p_grads = collect_grads(pool_dl, grad_model, proj_dim=args.gt_proj_dim,
                            adam_optimizer_state=None, gradient_type="sgd",
                            project_interval=args.project_interval)

    a_grads = normalize_embeddings_in_chunks(a_grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False)
    p_grads = normalize_embeddings_in_chunks(p_grads, chunk_size=10000, dim=1, eps=1e-12, in_place=False)
    gt_scores_eval = batch_cosine_similarity(
        dev_reps=a_grads, train_reps=p_grads, chunk_size=256, normalize=False,
    ).float().cpu().numpy()

    del grad_model, a_grads, p_grads, anchor_hf, pool_hf
    gc.collect()
    torch.cuda.empty_cache()

    enc_gt  = compute_native_metrics(gt_scores_eval, enc_pred,  agg_mode=args.agg_mode)
    base_gt = compute_native_metrics(gt_scores_eval, base_pred, agg_mode=args.agg_mode)

    tbl = [
        ["Semantic Baseline",
         f"{base_sketch['agg_spearman']:.4f}", f"{base_sketch['per_anchor_spearman_mean']:.4f}",
         f"{base_gt['agg_spearman']:.4f}",     f"{base_gt['per_anchor_spearman_mean']:.4f}"],
        ["Influence-Encoder",
         f"{enc_sketch['agg_spearman']:.4f}",  f"{enc_sketch['per_anchor_spearman_mean']:.4f}",
         f"{enc_gt['agg_spearman']:.4f}",       f"{enc_gt['per_anchor_spearman_mean']:.4f}"],
    ]
    print(tabulate(tbl,
                   headers=["Method", f"Agg ρ sketch(d={sketch_dim})", "PA ρ sketch",
                            f"Agg ρ GT(d={args.gt_proj_dim})", "PA ρ GT"],
                   tablefmt="github"))
    print(f"   sketch ≈ GT → CountSketch fine; gap is format/data-split.")
    print(f"   sketch >> GT → CountSketch still too lossy; raise INFLUCODER_PROJ_DIM further.")

    # Save
    combined_metrics = {
        "untrained": base_sketch,
        "trained": enc_sketch,
        "untrained_gt": base_gt,
        "trained_gt": enc_gt,
    }
    save_encoder_with_metadata(enc, args.output_dir, method_name="influence_encoder", args=args, metrics=combined_metrics)
