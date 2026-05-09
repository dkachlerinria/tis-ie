"""
LoGra Data Selection Pipeline
=============================
Computes LoGra-based influence scores between dev (test) and train data,
outputs a similarity matrix for use with selection.sim_subset.
Compatible with select_logra.sh and the tis-ie pipeline.
"""

import os
import sys
import torch
import json
import random
import numpy as np
import argparse
import warnings
import logging
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)
sys.path.append(os.path.join(script_dir, 'less'))

from less.utils.modeling_logra import LoGra

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

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
            example=x, tokenizer=tokenizer, max_seq_length=1024, include_response=True
        ),
        desc="Tokenizing training data"
    )
    logger.info(f"   ✓ Loaded {len(ds):,} training samples")
    return ds

def load_dev_dataset(dataset_name, tokenizer, n_samples=None, end_index=None):
    from common.data import construct_test_sample
    logger.info(f"📂 Loading dev dataset: {dataset_name}")

    if dataset_name.lower() == "bbh":
        # For BBH, we use the local JSONs if available, matching our eval pipeline
        eval_data_dir = os.environ.get("EVAL_DATA_DIR", "data/eval/bbh")
        # We'll just load the json files directly from the eval_data_dir
        raw_samples = []
        import glob
        task_files = glob.glob(os.path.join(eval_data_dir, "*.json"))
        for task_file in task_files:
            with open(task_file, "r") as f:
                data = json.load(f)
                for ex in data["examples"]:
                    raw_samples.append({
                        "prompts": ex["input"], 
                        "labels": ex["target"]
                    })
        
        from datasets import Dataset
        ds = Dataset.from_list(raw_samples)
    else:
        try:
            ds = load_dataset(dataset_name, split="test")
        except:
            ds = load_dataset(dataset_name, split="validation")
        
        def rename_keys(x):
            return {
                "prompts": x.get("prompt", x.get("input", "")),
                "labels": x.get("response", x.get("output", ""))
            }
        ds = ds.map(rename_keys)

    if end_index:
        ds = ds.select(range(min(end_index, len(ds))))
    if n_samples:
        random.seed(42)
        indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
        ds = ds.select(indices)

    # Use the EXACT same mapping as LESS eval
    ds = ds.map(
        lambda x: construct_test_sample(
            tokenizer=tokenizer, sample=x, max_length=1024
        ),
        desc="Tokenizing dev data"
    )
    logger.info(f"   ✓ Loaded {len(ds):,} dev samples")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LoGra influence scoring for data selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--ckpt_path', type=str, required=True,
        help='Path to warmup checkpoint (trained model directory)'
    )
    parser.add_argument(
        '--train_dataset_name', type=str, default="Harvard-DCML/tulu-v2-197K-processed",
        help='Training dataset name (HuggingFace dataset ID)'
    )
    parser.add_argument(
        '--dev_dataset_name', type=str, default="bbh",
        help='Dev/test dataset name (e.g., bbh, gsm8k)'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save similarity matrix'
    )
    parser.add_argument(
        '--end_index', type=int, default=None,
        help='Truncate training data at this index (for quick tests)'
    )
    parser.add_argument(
        '--n_train_samples', type=int, default=None,
        help='Sample N training examples (subset for quick tests)'
    )
    parser.add_argument(
        '--n_dev_samples', type=int, default=None,
        help='Sample N dev examples (subset for quick tests)'
    )
    parser.add_argument(
        '--rank', type=int, default=8,
        help='LoRA rank for LoGra'
    )
    parser.add_argument(
        '--mlp_only', action='store_true',
        help='Only apply LoRA to MLP layers'
    )
    parser.add_argument(
        '--grad_batch_size', type=int, default=1,
        help='Batch size for LoGra encoding'
    )
    parser.add_argument(
        '--seed', type=int, default=137
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 70)
    print("🔬 LoGra Influence Scoring Pipeline")
    print("=" * 70)
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Training Dataset: {args.train_dataset_name}")
    print(f"Dev Dataset: {args.dev_dataset_name}")
    print(f"LoGra Settings: Rank={args.rank}, MLP Only={args.mlp_only}")
    print("=" * 70)

    # Step 1: Initialize LoGra
    logger.info(f"\n🤖 Initializing LoGra from checkpoint: {args.ckpt_path}")
    logger.info("   (Loading model, this may take a minute...)")
    try:
        model = LoGra.from_pretrained(
            model_name=args.ckpt_path,
            rank=args.rank,
            mlp_only=args.mlp_only
        )
        logger.info(f"   ✓ LoGra model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load LoGra model: {e}")
        exit(1)

    # Step 2: Load data (needs model.tokenizer)
    logger.info("\n📥 Loading datasets...")
    train_dataset = load_train_dataset(args.train_dataset_name, model.tokenizer, args.n_train_samples, args.end_index)
    dev_dataset = load_dev_dataset(args.dev_dataset_name, model.tokenizer, args.n_dev_samples, args.end_index)

    # Step 3: Encode training pool (compute FIM + raw gradients)
    logger.info(f"\n🧮 Encoding {len(train_dataset)} training samples (compute FIM)...")
    logger.info(f"   (Processing with batch size {args.grad_batch_size}...)")
    train_embeds = model.encode(
        train_dataset,
        batch_size=args.grad_batch_size,
        is_test=False
    )
    logger.info(f"   ✓ Training embeddings shape: {train_embeds.shape}")

    # Step 4: Encode dev samples (apply FIM preconditioning)
    logger.info(f"\n🧮 Encoding {len(dev_dataset)} dev samples (apply FIM)...")
    logger.info(f"   (Processing {len(dev_dataset)} samples with batch size {args.grad_batch_size}...)")
    dev_embeds = model.encode(
        dev_dataset,
        batch_size=args.grad_batch_size,
        is_test=True
    )
    logger.info(f"   ✓ Dev embeddings shape: {dev_embeds.shape}")

    # Step 5: Compute similarity matrix (dev × train)
    logger.info(f"\n📊 Computing LoGra similarity matrix...")
    similarity_matrix = model.similarity(dev_embeds, train_embeds, mode='cosine')
    logger.info(f"   ✓ Similarity matrix shape: {similarity_matrix.shape}")

    # Step 6: Save similarity matrix
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dev_dataset_name}_cossim.npy")
    np.save(output_path, similarity_matrix)
    logger.info(f"\n✅ Similarity matrix saved to: {output_path}")

    # Save metadata
    metadata = {
        "method": "logra",
        "checkpoint": args.ckpt_path,
        "rank": args.rank,
        "mlp_only": args.mlp_only,
        "train_dataset": args.train_dataset_name,
        "dev_dataset": args.dev_dataset_name,
        "n_train": len(train_samples),
        "n_dev": len(dev_samples),
        "similarity_shape": similarity_matrix.shape,
    }
    metadata_path = os.path.join(args.output_dir, "logra_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📁 Metadata saved to: {metadata_path}")
    logger.info("\n" + "=" * 70)
