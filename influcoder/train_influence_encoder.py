"""
Influence-Encoder Training (Gradients Only)
==========================================
Trains an encoder on pre-stocked gradients and evaluates on:
  - Track 1: Native gradient similarity (Spearman correlation)
"""

import os
import sqlite3
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
    'small':  {'train_a': 1000,  'eval_a': 50,  'train_p': 2000, 'eval_p': 100,  'epochs': 4, 'max_train_pool': 10000},
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
# Database Loading
# =========================================================================

def load_all_doc_ids(db_path: str, seed: int = 42) -> list[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT p.doc_id FROM projections p
        INNER JOIN documents d ON d.doc_id = p.doc_id
        WHERE p.projection_seed = ? ORDER BY p.doc_id
    """, (seed,))
    doc_ids =[row[0] for row in cur.fetchall()]
    conn.close()
    return doc_ids

def load_stocked_samples_by_ids(db_path: str, doc_ids: list[str], seed: int = 42) -> tuple[list, np.ndarray]:
    if not doc_ids: return [], np.array([])
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT value FROM metadata WHERE key = 'proj_dim'")
        proj_dim = int(cur.fetchone()[0])
    except:
        raise ValueError("metadata 'proj_dim' missing – cannot infer gradient dtype")
    
    BATCH_SIZE = 500
    row_dict = {}
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(batch_ids))
        cur.execute(f"""
            SELECT d.doc_id, d.prompt, d.response, p.projected_gradient
            FROM documents d JOIN projections p ON d.doc_id = p.doc_id
            WHERE p.projection_seed = ? AND d.doc_id IN ({placeholders})
        """, (seed, *batch_ids))
        for row in cur.fetchall():
            row_dict[row[0]] = row
    conn.close()
    
    samples = []
    gradients =[]
    dtype = None
    
    for doc_id in doc_ids:
        if doc_id in row_dict:
            _, prompt, response, grad_blob = row_dict[doc_id]
            samples.append((prompt, response))
            
            if dtype is None:
                blob_len = len(grad_blob)
                if blob_len == proj_dim * 2: dtype = np.float16
                elif blob_len == proj_dim * 4: dtype = np.float32
                elif blob_len == proj_dim * 8: dtype = np.float64
                else:
                    raise ValueError(f"Cannot infer dtype for {doc_id}: blob_len={blob_len}, proj_dim={proj_dim}")

            grad_arr = np.frombuffer(grad_blob, dtype=dtype)
            if dtype == np.float16: grad_arr = grad_arr.astype(np.float32)
            gradients.append(grad_arr)
    
    return samples, np.array(gradients) if gradients else np.array([])

def samples_to_texts(samples, tokenizer=None):
    """Reconstruct chat-templated text from DB-stored (prompt, response) parts.

    gradient_stocking.render_for_storage splits the chat-templated full_text
    into a prefix (stored in `prompt`) and suffix (stored in `response`) such
    that prompt + response == full_text. So we simply concatenate to recover
    the exact text the gradient was computed against.
    """
    return [p + r for p, r in samples]

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
    """Hybrid loss: Pearson correlation + MSE.

    Optimising Pearson r directly matches the Spearman rank-correlation
    evaluation metric. MSE is added to provide a stable scale anchor.
    """

    def __init__(self, alpha=0.9):
        super().__init__()
        self.alpha = alpha

    def forward(self, scores, labels):
        s_flat = scores.view(-1)
        l_flat = labels.view(-1)
        s_cent = s_flat - s_flat.mean()
        l_cent = l_flat - l_flat.mean()
        cov = (s_cent * l_cent).sum()
        s_std = torch.sqrt((s_cent ** 2).sum() + 1e-8)
        l_std = torch.sqrt((l_cent ** 2).sum() + 1e-8)
        pearson_loss = 1.0 - (cov / (s_std * l_std))

        l_standardized = l_cent / (l_std / s_std.detach() + 1e-8) + s_flat.mean().detach()
        mse_loss = F.mse_loss(s_flat, l_standardized)
        return self.alpha * pearson_loss + (1 - self.alpha) * mse_loss

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
    parser.add_argument('--anchor_train_db', type=str, required=True)
    parser.add_argument('--anchor_eval_db', type=str, required=True)
    parser.add_argument('--pool_train_db', type=str, required=True)
    parser.add_argument('--pool_eval_db', type=str, required=True)
    parser.add_argument('--gradient_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_candidates_per_batch', type=int, default=20)
    parser.add_argument('--hard_ratio', type=float, default=0.2)
    parser.add_argument('--agg_mode', type=str, default='mean', choices=['mean', 'max'])
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.9, help="Weight for Pearson loss (1-alpha for MSE)")
    parser.add_argument('--output_dir', type=str, default="./checkpoints/influence_encoder")
    
    args = parser.parse_args()
    cfg = MODES[args.run_mode]
    epochs = args.epochs if args.epochs is not None else cfg['epochs']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(137)
    
    print("=" * 70)
    print(f"🔬 Influence-Encoder Training (Mode: {args.run_mode.upper()})")
    print("=" * 70)
    
    # 1. Load Data
    train_anchor_ids = load_all_doc_ids(args.anchor_train_db, args.gradient_seed)
    eval_anchor_ids  = load_all_doc_ids(args.anchor_eval_db,  args.gradient_seed)
    train_pool_ids   = load_all_doc_ids(args.pool_train_db,   args.gradient_seed)
    eval_pool_ids    = load_all_doc_ids(args.pool_eval_db,    args.gradient_seed)
    
    print(f"   📊 Database Summary (Seed {args.gradient_seed}):")
    print(f"      - Train Anchors: {len(train_anchor_ids)} IDs")
    print(f"      - Eval Anchors:  {len(eval_anchor_ids)} IDs")
    print(f"      - Train Pool:    {len(train_pool_ids)} IDs")
    print(f"      - Eval Pool:     {len(eval_pool_ids)} IDs")

    if not train_pool_ids:
        print(f"   ⚠️ WARNING: No IDs found in {args.pool_train_db}. Is the seed correct?")

    random.shuffle(train_anchor_ids)
    random.shuffle(eval_anchor_ids)
    random.shuffle(train_pool_ids)
    random.shuffle(eval_pool_ids)

    train_anchor_ids = train_anchor_ids[:cfg['train_a']]
    eval_anchor_ids = eval_anchor_ids[:cfg['eval_a']]
    train_pool_ids = train_pool_ids[:cfg['train_p']]
    eval_pool_ids = eval_pool_ids[:cfg['eval_p']]
    
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

    print("\n📥 Loading SQLite Data...")
    train_anchors, grads_train_anchors = load_stocked_samples_by_ids(args.anchor_train_db, train_anchor_ids, args.gradient_seed)
    eval_anchors, grads_eval_anchors = load_stocked_samples_by_ids(args.anchor_eval_db, eval_anchor_ids, args.gradient_seed)
    train_pool, grads_train_pool = load_stocked_samples_by_ids(args.pool_train_db, train_pool_ids, args.gradient_seed)
    eval_pool, grads_eval_pool = load_stocked_samples_by_ids(args.pool_eval_db, eval_pool_ids, args.gradient_seed)

    check_gradient_diagnostics("Train Anchors", grads_train_anchors)
    check_gradient_diagnostics("Eval Anchors",  grads_eval_anchors)
    check_gradient_diagnostics("Train Pool",    grads_train_pool)
    check_gradient_diagnostics("Eval Pool",     grads_eval_pool)

    train_anchors_text = samples_to_texts(train_anchors, gradient_tokenizer)
    eval_anchors_text  = samples_to_texts(eval_anchors,  gradient_tokenizer)
    train_text         = samples_to_texts(train_pool,    gradient_tokenizer)
    eval_text          = samples_to_texts(eval_pool,     gradient_tokenizer)
    
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
    loss_fn = InBatchLoss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    n_batches = (len(train_anchors_text) + args.batch_size - 1) // args.batch_size
    total_steps = math.ceil(n_batches * epochs / args.grad_accum_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    train_loss_avg = RunningAverage()
    global_batch_idx = 0

    for epoch in range(epochs):
        gen = AnchorBatchGenerator(train_anchors_text, train_text, true_scores_train,
                                   args.batch_size, args.n_candidates_per_batch,
                                   args.hard_ratio, seed=42+epoch)
        pbar = tqdm(gen, desc=f"Epoch {epoch+1}/{epochs}", total=n_batches)
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

            if global_batch_idx % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(ema_loss=f"{train_loss_avg.ema():.4f}")

        if global_batch_idx % args.grad_accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Quick Epoch Eval
        ev = quick_eval_native(enc, eval_anchors_text[:50], eval_text[:200], true_scores_eval[:50, :200], 
                               args.agg_mode, loss_fn, device, 
                               batch_size=args.batch_size, n_candidates=args.n_candidates_per_batch)
        print(f"\n   [Epoch {epoch+1}] Loss - Train: {train_loss_avg.ema():.4f} | Eval: {ev['loss']:.4f}")
        print(f"   [Epoch {epoch+1}] Eval Spearman - Agg ρ: {ev['agg_spearman']:.4f} | PA ρ: {ev['per_anchor_mean']:.4f}")

    # 3. Final Evaluation
    print("\n" + "=" * 70 + "\n🏁 FINAL EVALUATION\n" + "=" * 70)
    enc.eval()
    with torch.inference_mode():
        z_a = enc.encode(eval_anchors_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        z_e = enc.encode(eval_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        inf_metrics = compute_native_metrics(true_scores_eval, z_a @ z_e.T, agg_mode=args.agg_mode)
        
        # Baseline
        base_enc = SentenceTransformer(args.encoder_model, device=device)
        base_enc.max_seq_length = enc.max_seq_length
        zb_a = base_enc.encode(eval_anchors_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        zb_e = base_enc.encode(eval_text, normalize_embeddings=True, convert_to_numpy=True, batch_size=32)
        base_metrics = compute_native_metrics(true_scores_eval, zb_a @ zb_e.T, agg_mode=args.agg_mode)

    tbl = [
        ["Semantic Baseline", f"{base_metrics['agg_spearman']:.4f}", f"{base_metrics['per_anchor_spearman_mean']:.4f}"],
        ["Influence-Encoder", f"{inf_metrics['agg_spearman']:.4f}", f"{inf_metrics['per_anchor_spearman_mean']:.4f}"]
    ]
    print(tabulate(tbl, headers=["Method", "Aggregated ρ", "Per-Anchor ρ"], tablefmt="github"))

    # Save
    combined_metrics = {
        "untrained": base_metrics,
        "trained": inf_metrics
    }
    save_encoder_with_metadata(enc, args.output_dir, method_name="influence_encoder", args=args, metrics=combined_metrics)
