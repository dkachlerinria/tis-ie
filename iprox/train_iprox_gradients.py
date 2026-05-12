"""
IProX Training and Evaluation (Gradients)
==========================================
Trains a proxy model using IProX method and evaluates on:
  - Gradient similarity (Spearman correlation with true gradients)
  
This script:
1. Loads data from SQLite and converts to JSONL for IProX
2. Trains a compressed proxy model using IPSVD + gradient alignment
3. Computes proxy gradients and compares with true gradients
4. Reports the same metrics as Influence-Encoder for fair comparison
"""

import os
import sys
import shutil

# Fix IProX internal imports - add current folder and repo root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.append(script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import numpy as np
import argparse
import warnings
import logging
from typing import List, Tuple
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
)

# Import IProX utilities (raw from their codebase)
from iprox.utils.get_training_dataset import get_training_dataset
from iprox.utils.init_with_ipsvd import (
    init_proxy_model_with_IPSVD,
    load_proxy_model,
    save_proxy_model,
)
from iprox.utils.grad_align import train_with_gradient_alignment, get_target_layer_pairs
from iprox.utils.util import setseed

# Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# =========================================================================
# Run Modes Configuration (matching Influence-Encoder)
# =========================================================================
MODES = {
    'tiny':   {'train_a': 3,    'eval_a': 2,    'train_p': 5,    'eval_p': 3,    'epochs': 1},
    'quick':  {'train_a': 100,  'eval_a': 100,  'train_p': 200,  'eval_p': 200,  'epochs': 2},
    'medium': {'train_a': 2000, 'eval_a': 500,  'train_p': 6000, 'eval_p': 1000,  'epochs': 2},
    'full':   {'train_a': 4000, 'eval_a': 1000, 'train_p': 16000, 'eval_p': 4000, 'epochs': 2}
}

# =========================================================================
# Database Loading & Conversion
# =========================================================================

def load_all_doc_ids(db_path: str, seed: int = 42) -> list:
    """Load all document IDs from SQLite database"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT p.doc_id FROM projections p
        INNER JOIN documents d ON d.doc_id = p.doc_id
        WHERE p.projection_seed = ? ORDER BY p.doc_id
    """, (seed,))
    doc_ids = [row[0] for row in cur.fetchall()]
    conn.close()
    return doc_ids

def load_samples_by_ids(db_path: str, doc_ids: list) -> list:
    """Load (prompt, response) samples from SQLite"""
    if not doc_ids:
        return []
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    BATCH_SIZE = 500
    row_dict = {}
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(batch_ids))
        cur.execute(f"""
            SELECT doc_id, prompt, response
            FROM documents
            WHERE doc_id IN ({placeholders})
        """, batch_ids)
        for row in cur.fetchall():
            row_dict[row[0]] = (row[1], row[2])
    conn.close()
    
    samples = [row_dict[doc_id] for doc_id in doc_ids if doc_id in row_dict]
    return samples

def load_stocked_gradients(db_path: str, doc_ids: list, seed: int = 42) -> np.ndarray:
    """Load pre-computed (projected) gradients from SQLite"""
    if not doc_ids:
        return np.array([])
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    try:
        cur.execute("SELECT value FROM metadata WHERE key = 'proj_dim'")
        proj_dim = int(cur.fetchone()[0])
    except:
        raise ValueError("metadata 'proj_dim' missing – cannot load gradients")
    
    BATCH_SIZE = 500
    row_dict = {}
    for i in range(0, len(doc_ids), BATCH_SIZE):
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(batch_ids))
        cur.execute(f"""
            SELECT d.doc_id, p.projected_gradient
            FROM documents d JOIN projections p ON d.doc_id = p.doc_id
            WHERE p.projection_seed = ? AND d.doc_id IN ({placeholders})
        """, (seed, *batch_ids))
        for row in cur.fetchall():
            row_dict[row[0]] = row[1]
    conn.close()
    
    gradients = []
    dtype = None
    
    for doc_id in doc_ids:
        if doc_id in row_dict:
            grad_blob = row_dict[doc_id]
            
            if dtype is None:
                blob_len = len(grad_blob)
                if blob_len == proj_dim * 2:
                    dtype = np.float16
                elif blob_len == proj_dim * 4:
                    dtype = np.float32
                elif blob_len == proj_dim * 8:
                    dtype = np.float64
                else:
                    raise ValueError(f"Cannot infer dtype: blob_len={blob_len}, proj_dim={proj_dim}")
            
            grad_arr = np.frombuffer(grad_blob, dtype=dtype)
            if dtype == np.float16:
                grad_arr = grad_arr.astype(np.float32)
            gradients.append(grad_arr)
    
    return np.array(gradients) if gradients else np.array([])

def convert_to_jsonl(samples: list, output_path: str):
    """Convert (prompt, response) samples to JSONL format expected by IProX"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        for i, (prompt, response) in enumerate(samples):
            # IProX expects messages format
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            record = {
                "id": str(i),
                "messages": messages
            }
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"📝 Converted {len(samples)} samples to {output_path}")

# =========================================================================
# Gradient Computation with Proxy Model
# =========================================================================

def compute_sample_gradients(
    model: nn.Module,
    samples: list,
    tokenizer,
    max_seq_length: int,
    device: str,
    target_modules: list
) -> np.ndarray:
    """
    Compute per-sample gradients using the proxy model.
    Returns: [n_samples, gradient_dim] array
    """
    from common.data import encode_with_messages_format
    model.eval()  # Set to eval mode but we'll still compute gradients
    all_gradients = []
    
    for prompt, response in tqdm(samples, desc="Computing proxy gradients"):
        # Use standardized message formatting
        messages = [
            {"role": "user", "content": str(prompt).strip()},
            {"role": "assistant", "content": str(response).strip()}
        ]
        
        out = encode_with_messages_format(
            {"messages": messages}, 
            tokenizer, 
            max_seq_length=max_seq_length, 
            include_response=True
        )
        
        input_ids = out["input_ids"].unsqueeze(0).to(device)
        attention_mask = out["attention_mask"].unsqueeze(0).to(device)
        labels = out["labels"].unsqueeze(0).to(device)
        
        # Zero gradients
        model.zero_grad(set_to_none=True)
        
        # Forward + backward
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Collect gradients from target modules
            sample_grads = []
            for name, module in model.named_modules():
                is_target = any(name.endswith(tm) for tm in target_modules)
                if is_target:
                    # If it's an IProX LinearSVD layer, it has linear_A and linear_B
                    if hasattr(module, 'linear_A') and hasattr(module, 'linear_B'):
                        if module.linear_A.grad is not None and module.linear_B.grad is not None:
                            grad_A = module.linear_A.grad.detach().cpu().float().flatten().numpy()
                            grad_B = module.linear_B.grad.detach().cpu().float().flatten().numpy()
                            # Concatenate A and B proxy gradients
                            sample_grads.append(np.concatenate([grad_A, grad_B]))
                            
                    # If it's a standard layer (e.g., target model or skipped layer)
                    elif hasattr(module, 'weight') and module.weight.grad is not None:
                        sample_grads.append(module.weight.grad.detach().cpu().float().flatten().numpy())
            
            if sample_grads:
                concat_grad = np.concatenate(sample_grads)
                all_gradients.append(concat_grad)
            else:
                logger.warning(f"No gradients collected for sample")
                # Add zero gradient as fallback
                all_gradients.append(np.zeros(1))
        
        except Exception as e:
            logger.warning(f"Error computing gradient for sample: {e}")
            # Add zero gradient
            if len(all_gradients) > 0:
                all_gradients.append(np.zeros_like(all_gradients[0]))
            else:
                all_gradients.append(np.zeros(1))
    
    if all_gradients:
        # Ensure all gradients have same dimension
        max_dim = max(g.shape[0] for g in all_gradients)
        padded_grads = []
        for g in all_gradients:
            if g.shape[0] < max_dim:
                g = np.pad(g, (0, max_dim - g.shape[0]))
            padded_grads.append(g)
        return np.array(padded_grads)
    else:
        return np.array([])

# =========================================================================
# Evaluation Metrics (same as Influence-Encoder)
# =========================================================================

def safe_normalize(arr, eps=1e-8):
    """Normalize numpy array along last dimension"""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms

def compute_native_metrics(true_scores, pred_scores, agg_mode='mean'):
    """Compute Spearman correlations (same logic as Influence-Encoder)"""
    if true_scores.size == 0 or pred_scores.size == 0:
        return {'agg_spearman': 0.0, 'per_anchor_spearman_mean': 0.0}
    
    true_scores = np.nan_to_num(true_scores, nan=0.0)
    pred_scores = np.nan_to_num(pred_scores, nan=0.0)
    
    # Aggregated scores
    if agg_mode == 'mean':
        true_agg = true_scores.mean(axis=0)
        pred_agg = pred_scores.mean(axis=0)
    else:
        true_agg = true_scores.max(axis=0)
        pred_agg = pred_scores.max(axis=0)
    
    if true_agg.size < 2 or np.std(true_agg) < 1e-9 or np.std(pred_agg) < 1e-9:
        agg_spearman = 0.0
    else:
        agg_spearman, _ = spearmanr(true_agg, pred_agg)
    
    # Per-anchor scores
    per_anchor_spearman = []
    n_anchors = true_scores.shape[0]
    for i in range(n_anchors):
        if np.std(true_scores[i]) < 1e-9 or np.std(pred_scores[i]) < 1e-9:
            continue
        corr, _ = spearmanr(true_scores[i], pred_scores[i])
        if not np.isnan(corr):
            per_anchor_spearman.append(corr)
    
    return {
        'agg_spearman': float(agg_spearman),
        'per_anchor_spearman_mean': float(np.mean(per_anchor_spearman)) if per_anchor_spearman else 0.0,
    }

def iprox_quick_eval(proxy_model, eval_anchors, eval_pool, true_scores_eval, tokenizer, args, device):
    """Quick validation signal using 20x20 subset (same logic as Influence-Encoder)"""
    # Use only first 20 anchors and 20 pool samples
    a_subset = eval_anchors[:20]
    p_subset = eval_pool[:20]
    s_subset = true_scores_eval[:20, :20]
    
    proxy_model.eval()
    # We MUST NOT use torch.no_grad() here because compute_sample_gradients 
    # needs to build the graph to call loss.backward()
    
    # Ensure parameters have requires_grad=True
    for p in proxy_model.parameters():
        p.requires_grad_(True)
        
    ga = compute_sample_gradients(proxy_model, a_subset, tokenizer, args.max_seq_length, device, args.target_modules)
    gp = compute_sample_gradients(proxy_model, p_subset, tokenizer, args.max_seq_length, device, args.target_modules)
        
    if ga.size == 0 or gp.size == 0:
        proxy_model.train()
        return {'Agg ρ': 0.0, 'PA ρ': 0.0}
        
    scores = safe_normalize(ga) @ safe_normalize(gp).T
    metrics = compute_native_metrics(s_subset, scores, agg_mode=args.agg_mode)
    proxy_model.train()
    return {'Agg ρ': metrics['agg_spearman'], 'PA ρ': metrics['per_anchor_spearman_mean']}

# =========================================================================
# Save Proxy Model with Metadata
# =========================================================================

def save_iprox_with_metadata(proxy_model, target_model, tokenizer, output_dir, args):
    """Save IProX proxy model with standardized structure"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save proxy model checkpoint
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    final_checkpoint = os.path.join(output_dir, "final_pytorch_model.bin")
    if os.path.exists(final_checkpoint):
        # Copy to standardized location
        shutil.copy(final_checkpoint, os.path.join(model_dir, "pytorch_model.bin"))
    
    # Save config and tokenizer for easy loading
    target_model.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save metadata
    metadata = {
        "method": "iprox",
        "target_model": args.target_model,
        "sparsity": args.sparsity,
        "init_method": args.init_method,
        "gradient_seed": args.gradient_seed,
        "run_mode": args.run_mode,
        "lambda_anchor": args.lambda_anchor,
        "target_modules": args.target_modules,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ IProX model saved to {output_dir}")
    print(f"   - Model: {model_dir}")
    print(f"   - Metadata: metadata.json")

# =========================================================================
# Main Execution
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train IProX proxy and evaluate on gradient similarity')
    
    # Data args
    parser.add_argument('--run_mode', type=str, default="quick", choices=['tiny', 'quick', 'medium', 'full'])
    parser.add_argument('--anchor_train_db', type=str, required=True, help='Path to train anchor SQLite database')
    parser.add_argument('--anchor_eval_db', type=str, required=True, help='Path to eval anchor SQLite database')
    parser.add_argument('--pool_train_db', type=str, required=True, help='Path to train pool SQLite database')
    parser.add_argument('--pool_eval_db', type=str, required=True, help='Path to eval pool SQLite database')
    parser.add_argument('--gradient_seed', type=int, default=42)
    parser.add_argument('--agg_mode', type=str, default='mean', choices=['mean', 'max'])
    
    # Model args
    parser.add_argument('--target_model', type=str, default="Qwen/Qwen3-0.6B-Base",
                       help="Target LLM (must match the model used to generate gradients in DB)")
    parser.add_argument('--sparsity', type=float, default=0.5, help='SVD pruning sparsity')
    parser.add_argument('--init_method', type=str, default='IPSVD', choices=['RANDOM', 'SVD', 'IPSVD'])
    parser.add_argument('--target_modules', nargs='+',
                       default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=None, help='Override mode default epochs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_anchor', type=float, default=0.5, help='KD loss weight')
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # Output args
    parser.add_argument('--output_dir', type=str, default="./checkpoints/iprox")
    parser.add_argument('--temp_data_dir', type=str, default="./temp_iprox_data")
    parser.add_argument('--seed', type=int, default=137)
    
    args = parser.parse_args()
    cfg = MODES[args.run_mode]
    if args.epochs is None:
        args.epochs = cfg['epochs']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setseed(args.seed)
    
    print("=" * 70)
    print(f"🔬 IProX Training (Mode: {args.run_mode.upper()})")
    print("=" * 70)
    print(f"Target Model: {args.target_model}")
    print(f"Sparsity: {args.sparsity}, Init: {args.init_method}, Epochs: {args.epochs}")
    print("=" * 70)
    
    # =====================================================================
    # Step 1: Load Data from SQLite
    # =====================================================================
    logger.info("📥 Loading data from SQLite databases...")
    
    train_anchor_ids = load_all_doc_ids(args.anchor_train_db, args.gradient_seed)
    eval_anchor_ids = load_all_doc_ids(args.anchor_eval_db, args.gradient_seed)
    train_pool_ids = load_all_doc_ids(args.pool_train_db, args.gradient_seed)
    eval_pool_ids = load_all_doc_ids(args.pool_eval_db, args.gradient_seed)
    
    random.shuffle(train_anchor_ids)
    random.shuffle(eval_anchor_ids)
    random.shuffle(train_pool_ids)
    random.shuffle(eval_pool_ids)
    
    # Split according to mode
    train_anchor_ids = train_anchor_ids[:cfg['train_a']]
    eval_anchor_ids = eval_anchor_ids[:cfg['eval_a']]
    train_pool_ids = train_pool_ids[:cfg['train_p']]
    eval_pool_ids = eval_pool_ids[:cfg['eval_p']]
    
    logger.info(f"Train: {len(train_anchor_ids)} anchors, {len(train_pool_ids)} pool")
    logger.info(f"Eval:  {len(eval_anchor_ids)} anchors, {len(eval_pool_ids)} pool")
    
    # Load text samples
    train_anchors = load_samples_by_ids(args.anchor_train_db, train_anchor_ids)
    train_pool = load_samples_by_ids(args.pool_train_db, train_pool_ids)
    eval_anchors = load_samples_by_ids(args.anchor_eval_db, eval_anchor_ids)
    eval_pool = load_samples_by_ids(args.pool_eval_db, eval_pool_ids)
    
    # Load true gradients for evaluation
    logger.info("📊 Loading true gradients for evaluation...")
    true_grads_eval_anchors = load_stocked_gradients(args.anchor_eval_db, eval_anchor_ids, args.gradient_seed)
    true_grads_eval_pool = load_stocked_gradients(args.pool_eval_db, eval_pool_ids, args.gradient_seed)
    
    # Compute true similarity matrix
    true_grads_eval_anchors_norm = safe_normalize(true_grads_eval_anchors)
    true_grads_eval_pool_norm = safe_normalize(true_grads_eval_pool)
    true_scores_eval = true_grads_eval_anchors_norm @ true_grads_eval_pool_norm.T
    
    logger.info(f"True scores matrix shape: {true_scores_eval.shape}")
    
    # =====================================================================
    # Step 2: Convert to JSONL for IProX
    # =====================================================================
    logger.info("📝 Converting data to JSONL format...")
    os.makedirs(args.temp_data_dir, exist_ok=True)
    
    # Combine all training data
    all_train_samples = train_anchors + train_pool
    train_jsonl = os.path.join(args.temp_data_dir, "train_combined.jsonl")
    convert_to_jsonl(all_train_samples, train_jsonl)
    
    # Prepare small eval set for IPSVD initialization
    eval_init_samples = eval_anchors[:min(50, len(eval_anchors))] + eval_pool[:min(50, len(eval_pool))]
    eval_jsonl = os.path.join(args.temp_data_dir, "eval_init.jsonl")
    convert_to_jsonl(eval_init_samples, eval_jsonl)
    
    # =====================================================================
    # Step 3: Load Tokenizer and Target Model
    # =====================================================================
    logger.info(f"🤖 Loading target model: {args.target_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Resize embeddings if needed
    if len(tokenizer) > target_model.get_input_embeddings().weight.shape[0]:
        logger.info(f"Resizing embeddings from {target_model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        target_model.resize_token_embeddings(len(tokenizer))
    
    # =====================================================================
    # Step 4: Prepare Datasets
    # =====================================================================
    logger.info("📦 Preparing datasets...")
    
    train_dataset = get_training_dataset(
        [train_jsonl],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        sample_percentage=1.0,
        seed=args.seed
    )
    
    eval_dataset = get_training_dataset(
        [eval_jsonl],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        sample_percentage=1.0,
        seed=args.seed
    )
    
    # Remove extra columns
    cols_to_remove = [c for c in ["dataset", "id", "messages"] if c in train_dataset.features]
    if cols_to_remove:
        train_dataset = train_dataset.remove_columns(cols_to_remove)
        eval_dataset = eval_dataset.remove_columns(cols_to_remove)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    # Split for IPSVD
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=args.max_seq_length
    )
    
    # DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # =====================================================================
    # Step 5: Initialize Proxy Model with IPSVD
    # =====================================================================
    model_slug = os.path.basename(args.target_model.rstrip('/\\')) or args.target_model.replace('/', '_').replace('\\', '_').replace(':', '_')
    output_dir = os.path.join(
        args.output_dir,
        f"proxy_{model_slug}",
        f"sp{args.sparsity}_lam{args.lambda_anchor}_lr{args.lr:.0e}_ep{args.epochs}"
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"💾 Output directory: {output_dir}")
    
    checkpoint_path = os.path.join(output_dir, "init_pytorch_model.bin")
    
    if os.path.exists(checkpoint_path):
        logger.info(f"✓ Found existing init checkpoint, loading...")
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method="RANDOM",
            freeze_non_md_param=True,
            target_modules=args.target_modules,
            min_rank_multiple=1,  # <--- FIX 1: Prevent SVD layers from being skipped
        )
        load_proxy_model(proxy_model, checkpoint_path)
    else:
        logger.info(f"🔧 Initializing proxy with {args.init_method}...")
        proxy_model = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method=args.init_method,
            freeze_non_md_param=True,
            target_modules=args.target_modules,
            min_rank_multiple=1,  # <--- FIX 1: Prevent SVD layers from being skipped
        )
        save_proxy_model(proxy_model, checkpoint_path)
        logger.info(f"💾 Saved initialized proxy to {checkpoint_path}")
    
    # =====================================================================
    # Step 6: Get Layer Pairs and Setup Training
    # =====================================================================
    logger.info("🔗 Getting target-proxy layer pairs...")
    
    # IPSVD wraps the original Linear so the LinearSVD lives at "<name>.base_layer".
    # We pair the target's Linear with the proxy's nested LinearSVD.
    layer_pairs = []
    proxy_dict = dict(proxy_model.named_modules())

    for name, t_mod in target_model.named_modules():
        if not any(name.endswith(tm) for tm in args.target_modules):
            continue
        matched = False
        for proxy_path in (name, f"{name}.base_layer"):
            cand = proxy_dict.get(proxy_path)
            if cand is not None and hasattr(cand, "linear_A") and hasattr(cand, "linear_B"):
                layer_pairs.append((t_mod, cand))
                matched = True
                break
        if not matched:
            logger.warning(f"Layer {name}: no LinearSVD found in proxy (skipped).")

    if not layer_pairs:
        logger.error("❌ No matching layer pairs found!")
        exit(1)
    
    logger.info(f"✓ Found {len(layer_pairs)} layer pairs for gradient alignment")
    
    # Freeze target model except for layers we'll track
    for p in target_model.parameters():
        p.requires_grad_(False)
    
    for t_layer, _ in layer_pairs:
        t_layer.weight.requires_grad_(True)
        if hasattr(t_layer, 'bias') and t_layer.bias is not None:
            t_layer.bias.requires_grad_(True)
    
    # Optimizer for proxy
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, proxy_model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    # Initial baseline evaluation
    logger.info("\n🔍 Running initial evaluation (before training baseline)...")
    init_metrics = iprox_quick_eval(proxy_model, eval_anchors, eval_pool, true_scores_eval, tokenizer, args, device)
    logger.info(f"Initial Eval - Agg ρ: {init_metrics['Agg ρ']:.4f} | PA ρ: {init_metrics['PA ρ']:.4f}\n")

    #
    
    # =====================================================================
    # Step 7: Train with Gradient Alignment (IProX method)
    # =====================================================================
    logger.info("🚀 Starting gradient alignment training...")
    
    train_with_gradient_alignment(
        target_model=target_model,
        proxy_model=proxy_model,
        layer_pairs=layer_pairs,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_path=output_dir,
        lambda_anchor=args.lambda_anchor,
        max_steps=-1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_interval=50,
        eval_interval=50,
        epoch_callback=lambda m: iprox_quick_eval(
            m, eval_anchors, eval_pool, true_scores_eval, tokenizer, args, device
        )
    )
    
    # =====================================================================
    # Step 8: Final Evaluation & Saving
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("🏁 FINAL EVALUATION")
    logger.info("=" * 70)
    
    # Load the final trained proxy
    final_checkpoint = os.path.join(output_dir, "final_pytorch_model.bin")
    if os.path.exists(final_checkpoint):
        logger.info(f"Loading final model from {final_checkpoint}")
        proxy_model_eval = init_proxy_model_with_IPSVD(
            base_model=target_model,
            loader_src=val_dataloader,
            sparsity=args.sparsity,
            init_method="RANDOM",
            freeze_non_md_param=False,  # All params for inference
            target_modules=args.target_modules,
            min_rank_multiple=1,  # <--- FIX 1: Prevent SVD layers from being skipped
        )
        load_proxy_model(proxy_model_eval, final_checkpoint)
    else:
        logger.warning("⚠️  Final checkpoint not found, using current proxy_model")
        proxy_model_eval = proxy_model
    
    import gc
    # Compute proxy gradients incrementally to avoid CPU OOM
    logger.info("🧮 Computing proxy gradients and similarity matrix incrementally...")
    
    # 1. Compute anchor gradients and normalize them
    # (Anchor set is usually smaller, e.g., 200 samples)
    n_anchors = len(eval_anchors)
    n_pool = len(eval_pool)
    anchor_matrix = None 
    
    for i, anchor in enumerate(tqdm(eval_anchors, desc="Encoding anchors")):
        grad = compute_sample_gradients(
            model=proxy_model_eval,
            samples=[anchor],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            device=device,
            target_modules=args.target_modules
        )
        if grad.size == 0: continue
        
        grad_norm = safe_normalize(grad)
        
        if anchor_matrix is None:
            # Lazy init to get the gradient dimension
            anchor_matrix = np.zeros((n_anchors, grad.shape[1]), dtype=np.float32)
        
        anchor_matrix[i] = grad_norm[0]
        
        if i % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # 2. Compute pool gradients one-by-one and dot product
    proxy_scores_eval = np.zeros((n_anchors, n_pool), dtype=np.float32)
    
    for j, pool_sample in enumerate(tqdm(eval_pool, desc="Encoding pool (streaming similarity)")):
        grad_p = compute_sample_gradients(
            model=proxy_model_eval,
            samples=[pool_sample],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            device=device,
            target_modules=args.target_modules
        )
        if grad_p.size == 0: continue
        
        grad_p_norm = safe_normalize(grad_p)
        
        if anchor_matrix is not None:
            # Dot product against all anchors
            proxy_scores_eval[:, j] = anchor_matrix @ grad_p_norm[0]
            
        if j % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Clean up large matrix
    if anchor_matrix is not None:
        del anchor_matrix
    gc.collect()
    
    # Compute metrics
    metrics = compute_native_metrics(true_scores_eval, proxy_scores_eval, agg_mode=args.agg_mode)
    
    # Display results (same format as Influence-Encoder)
    print("\n" + "=" * 70)
    print("RESULTS (Gradient Similarity Prediction)")
    print("=" * 70)
    tbl = [
        ["IProX Proxy", f"{metrics['agg_spearman']:.4f}", f"{metrics['per_anchor_spearman_mean']:.4f}"]
    ]
    print(tabulate(tbl, headers=["Method", "Aggregated ρ", "Per-Anchor ρ"], tablefmt="github"))
    print("=" * 70)

    # Save finalized model with metadata
    save_iprox_with_metadata(
        proxy_model=proxy_model_eval,
        target_model=target_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        args=args
    )
    
    logger.info(f"\n✅ Training and evaluation complete!")
    logger.info(f"📁 Model saved to: {output_dir}")