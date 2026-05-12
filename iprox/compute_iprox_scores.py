import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import logging
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add repo root and iprox folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==================== HELPERS ====================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_train_dataset(dataset_name, tokenizer, n_samples=None, end_index=None):
    from common.data import encode_with_messages_format
    logger.info(f"📂 Loading training dataset: {dataset_name}")
    
    if os.path.exists(dataset_name):
        ds = load_dataset("json", data_files=[dataset_name])["train"]
    else:
        ds = load_dataset(dataset_name, split="train")

    if end_index:
        ds = ds.select(range(min(end_index, len(ds))))
    
    if n_samples:
        random.seed(42)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)

    # Use the EXACT same mapping as LESS
    ds = ds.map(
        lambda x: encode_with_messages_format(
            example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True
        ),
        desc="Tokenizing training data"
    )
    logger.info(f"   ✓ Loaded {len(ds):,} training samples")
    return ds

def load_dev_dataset(dataset_name, tokenizer, n_samples=None, end_index=None):
    from common.data import construct_test_sample
    logger.info(f"📂 Loading dev dataset: {dataset_name}")

    # Match LESS pipeline: Use the targeted query set for influence estimation
    try:
        ds = load_dataset(
            "Harvard-DCML/targeted-query-set-processed",
            dataset_name,
            split="dev",
        )
    except Exception as e:
        logger.warning(f"   ⚠️ Could not load from targeted-query-set: {e}. Falling back to default splits.")
        try:
            ds = load_dataset(dataset_name, split="test")
        except:
            ds = load_dataset(dataset_name, split="validation")
    
    # Ensure standard keys for the mapping
    def rename_keys(x):
        return {
            "prompts": x.get("prompt", x.get("input", x.get("question", ""))),
            "labels": x.get("response", x.get("output", x.get("answer", "")))
        }
    ds = ds.map(rename_keys)

    if len(ds) == 0:
        logger.error(f"❌ Error: Dataset '{dataset_name}' is empty.")
        exit(1)

    if end_index:
        ds = ds.select(range(min(end_index, len(ds))))
    if n_samples:
        random.seed(42)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)

    # Use the EXACT same mapping as LESS eval
    ds = ds.map(
        lambda x: construct_test_sample(
            tokenizer=tokenizer, sample=x, max_length=2048
        ),
        desc="Tokenizing dev data"
    )
    logger.info(f"   ✓ Loaded {len(ds):,} dev samples")
    return ds

# ==================== IPROX SCORING LOGIC ====================

def compute_proxy_gradient(model, batch, device, target_modules):
    model.zero_grad(set_to_none=True)
    
    # Send to device and add batch dimension
    input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(device)
    labels = torch.tensor(batch["labels"]).unsqueeze(0).to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    sample_grads = []
    for name, module in model.named_modules():
        if any(name.endswith(tm) for tm in target_modules):
            if hasattr(module, 'linear_A') and hasattr(module, 'linear_B'):
                if module.linear_A.grad is not None and module.linear_B.grad is not None:
                    grad_A = module.linear_A.grad.detach().cpu().float().flatten().numpy()
                    grad_B = module.linear_B.grad.detach().cpu().float().flatten().numpy()
                    sample_grads.append(np.concatenate([grad_A, grad_B]))
    
    if not sample_grads:
        return None
    return np.concatenate(sample_grads)

def safe_normalize(grad, eps=1e-8):
    norm = np.linalg.norm(grad)
    return grad / max(norm, eps)

def main():
    parser = argparse.ArgumentParser(description='IProX similarity matrix computation')
    parser.add_argument("--proxy_path", type=str, required=True, help="Path to trained proxy model directory")
    parser.add_argument("--train_dataset_name", type=str, default="Harvard-DCML/tulu-v2-197K-processed")
    parser.add_argument("--dev_dataset_name", type=str, default="bbh")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--n_train_samples", type=int, default=None)
    parser.add_argument("--n_dev_samples", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--target_modules", nargs="+", default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🔬 IProX Influence Scoring Pipeline")
    print("=" * 70)
    print(f"Proxy Path: {args.proxy_path}")
    print(f"Training Dataset: {args.train_dataset_name}")
    print(f"Dev Dataset: {args.dev_dataset_name}")
    print("=" * 70)

    # 1. Load Model & Tokenizer
    logger.info(f"🤖 Loading Proxy Model: {args.proxy_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.proxy_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Expand "all-linear" if specified
    target_modules = args.target_modules
    if "all-linear" in target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Re-initialize the LinearSVD structure
    proxy_model = init_proxy_model_with_IPSVD(
        base_model=base_model,
        loader_src=None,
        sparsity=args.sparsity,
        init_method="RANDOM",
        target_modules=target_modules,
        min_rank_multiple=1
    )
    
    # Load the trained weights
    final_bin = os.path.join(args.proxy_path, "pytorch_model.bin")
    if not os.path.exists(final_bin):
        # Fallback for alternative paths
        final_bin = os.path.join(os.path.dirname(args.proxy_path), "final_pytorch_model.bin")
    
    load_proxy_model(proxy_model, final_bin)
    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    # 2. Load Datasets
    logger.info("\n📥 Loading datasets...")
    train_dataset = load_train_dataset(args.train_dataset_name, tokenizer, args.n_train_samples, args.end_index)
    dev_dataset = load_dev_dataset(args.dev_dataset_name, tokenizer, args.n_dev_samples, args.end_index)

    # 3. Compute Dev (Anchor) Gradients
    logger.info(f"\n🧮 Computing gradients for {len(dev_dataset)} dev samples...")
    dev_grads = []
    for i in tqdm(range(len(dev_dataset)), desc="Dev Gradients"):
        g = compute_proxy_gradient(proxy_model, dev_dataset[i], device, args.target_modules)
        if g is not None:
            dev_grads.append(safe_normalize(g))
        else:
            dev_grads.append(np.zeros(1))
    
    dev_matrix = np.stack(dev_grads) # [n_dev, dim]
    
    # 4. Compute Train Gradients and Similarity
    logger.info(f"\n📊 Computing similarity matrix for {len(train_dataset)} training samples...")
    similarity_matrix = np.zeros((len(dev_dataset), len(train_dataset)), dtype=np.float32)
    
    for j in tqdm(range(len(train_dataset)), desc="Train Similarity"):
        g_t = compute_proxy_gradient(proxy_model, train_dataset[j], device, args.target_modules)
        if g_t is not None:
            g_t_norm = safe_normalize(g_t)
            similarity_matrix[:, j] = dev_matrix @ g_t_norm
        
        if j % 1000 == 0:
            torch.cuda.empty_cache()

    # 5. Save Result
    output_path = os.path.join(args.output_dir, f"{args.dev_dataset_name}_cossim.npy")
    np.save(output_path, similarity_matrix)
    logger.info(f"\n✅ Similarity matrix saved to: {output_path}")

    # Save metadata
    import json
    metadata = {
        "method": "iprox",
        "proxy_path": args.proxy_path,
        "train_dataset": args.train_dataset_name,
        "dev_dataset": args.dev_dataset_name,
        "n_train": len(train_dataset),
        "n_dev": len(dev_dataset),
        "similarity_shape": similarity_matrix.shape,
    }
    with open(os.path.join(args.output_dir, "iprox_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
