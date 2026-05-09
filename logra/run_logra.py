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

def format_for_logra(samples):
    """Format samples as dicts for LoGra encoding."""
    return [{"input": p, "output": r} for p, r in samples]

def load_train_data(dataset_name, n_samples=None, end_index=None):
    """Load training dataset from HuggingFace."""
    logger.info(f"📂 Loading training dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    if end_index:
        ds = ds.select(range(min(end_index, len(ds))))

    samples = []
    for item in ds:
        # Handle Tulu messages format
        if "messages" in item and len(item["messages"]) >= 2:
            msgs = item["messages"]
            if msgs[-1]["role"] == "assistant":
                response = msgs[-1]["content"]
                prompt = msgs[:-1] # List of messages
                samples.append({"prompt": prompt, "response": response, "pre_formatted": True})
        else:
            p = item.get("prompt", item.get("input", item.get("instruction", "")))
            r = item.get("response", item.get("output", ""))
            if p and r:
                samples.append({"prompt": p, "response": r, "pre_formatted": False})

    if n_samples:
        random.seed(42)
        samples = random.sample(samples, min(n_samples, len(samples)))

    logger.info(f"   ✓ Loaded {len(samples):,} training samples")
    return samples

def load_dev_data(dataset_name, n_samples=None, end_index=None):
    """Load development/test dataset (e.g., BBH benchmark)."""
    logger.info(f"📂 Loading dev dataset: {dataset_name}")

    if dataset_name.lower() == "bbh":
        bbh_tasks = [
            'boolean_expressions', 'causal_judgement', 'date_understanding',
            'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
            'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
            'logical_deduction_seven_objects', 'logical_deduction_three_objects',
            'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
            'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
            'ruin_names', 'salient_translation_error_detection', 'snarks',
            'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects',
            'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects',
            'web_of_lies', 'word_sorting'
        ]
        samples = []
        for task in bbh_tasks:
            try:
                # Load BBH with the same logic as our evaluation pipeline
                ds = load_dataset("lukaemon/bbh", task, split="test")
                for item in ds:
                    p = item.get("input", "")
                    r = item.get("target", "")
                    if p and r:
                        # Mark as bbh to trigger A: suffix in LoGraModel
                        samples.append({"prompt": p, "response": r, "dataset": "bbh", "pre_formatted": True})
            except Exception as e:
                logger.warning(f"Failed to load BBH task {task}: {e}")
    else:
        try:
            ds = load_dataset(dataset_name, split="test")
        except:
            ds = load_dataset(dataset_name, split="validation")
        
        samples = []
        for item in ds:
            p = item.get("prompt", item.get("input", item.get("instruction", "")))
            r = item.get("response", item.get("output", ""))
            if p and r:
                samples.append({"prompt": p, "response": r, "pre_formatted": True})

    if end_index:
        samples = samples[:end_index]
    if n_samples:
        random.seed(42)
        samples = random.sample(samples, min(n_samples, len(samples)))

    logger.info(f"   ✓ Loaded {len(samples):,} dev samples")
    return samples


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

    # Step 1: Load data
    logger.info("\n📥 Loading datasets...")
    train_samples = load_train_data(args.train_dataset_name, args.n_train_samples, args.end_index)
    dev_samples = load_dev_data(args.dev_dataset_name, args.n_dev_samples, args.end_index)

    if not train_samples or not dev_samples:
        logger.error("Failed to load data")
        exit(1)

    # Step 2: Initialize LoGra
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

    # Step 3: Encode training pool (compute FIM + raw gradients)
    logger.info(f"\n🧮 Encoding {len(train_samples)} training samples (compute FIM)...")
    logger.info(f"   (Processing {len(train_samples)} samples with batch size {args.grad_batch_size}...)")
    train_embeds = model.encode(
        train_samples,
        batch_size=args.grad_batch_size,
        is_test=False
    )
    logger.info(f"   ✓ Training embeddings shape: {train_embeds.shape}")

    # Step 4: Encode dev samples (apply FIM preconditioning)
    logger.info(f"\n🧮 Encoding {len(dev_samples)} dev samples (apply FIM)...")
    logger.info(f"   (Processing {len(dev_samples)} samples with batch size {args.grad_batch_size}...)")
    dev_embeds = model.encode(
        dev_samples,
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
