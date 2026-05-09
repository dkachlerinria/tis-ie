"""
Gradient Stocking Script (AirRep Aligned + LoRA Warmup)
======================================================
COMPATIBLE WITH TRAINING SCRIPT - Creates databases that work for both anchor and pool roles.
Includes Dolci, Platinum, & MMLU dataset support while maintaining original text format constraints.
"""

import os
import json
import argparse
import sqlite3
import hashlib
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, VerificationMode
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from formatting import ensure_chat_template, format_sample, render_for_storage
from torch.nn.utils.rnn import pad_sequence

# ==================== DEFAULTS ====================
DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
DEFAULT_PROJ_DIM = 131072
DEFAULT_SEEDS = [42, 137]
GRAD_BATCH_SIZE = 4
MAX_SEQ_LEN = 1024

# Warmup parameters
WARMUP_LR = 2e-5       # Aligned with paper's 2e-5
WARMUP_EPOCHS = 2      # Aligned with paper's 2 epochs
WARMUP_BATCH_SIZE = 4  # Aligned with paper's batch size of 32
# ==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================================================================
# 1. Formatting & Masking Builders
# =========================================================================

def format_item(item):
    """Specific to FLAN's AirRep format."""
    prompt = ''
    response = ''

    if 'data' in item:
        prompt = item['data'][0]
        prompt = " ".join(prompt.split()[:100])
        response = item['data'][1]

    if 'input' in item:
        prompt = item.get('instruction', '') + ' ' + item['input']

    for op in ['inputs', 'prompt', 'instruction']:
        if op in item:
            prompt = item[op]
            break

    for op in ['response', 'targets', 'output']:
        if op in item:
            response = item[op]
            break

    prompt = 'Question: ' + " ".join(prompt.split()[:256]) + '\nAnswer:'
    return prompt, response


def build_example(sample, tokenizer, dataset_name=None):
    """Chat-template formatted; dispatched per dataset.

    For FLAN/dolci/platinum: 2-turn instruction. For MMLU: lm-eval-style MC
    body in user turn + answer letter in assistant turn. Same format used
    everywhere this dataset appears in the pipeline.
    """
    out = format_sample(sample, dataset_name or "instruction", tokenizer, MAX_SEQ_LEN)
    return out["input_ids"], out["labels"]

# =========================================================================
# 2. Dataset Loading & Partitioning
# =========================================================================

def load_raw_dataset(dataset_name, tokenizer, n_samples=None, split=None):
    """Loads and filters raw datasets before partitioning."""
    processed = []
    
    if dataset_name == "dolci":
        print("📂 Loading tasksource/dolci-instruct...")
        ds = load_dataset("tasksource/dolci-instruct", split="train")
        for row in tqdm(ds, desc="Filtering Dolci"):
            prompt = str(row.get("prompt", ""))
            response = str(row.get("answer", ""))
            if not response.strip() or len(prompt.strip()) < 10:
                continue
            processed.append({"prompt": prompt, "response": response})

    elif dataset_name == "platinum":
        print("📂 Loading madrylab/platinum-bench...")
        configs = get_dataset_config_names("madrylab/platinum-bench")
        vqa_keywords = ["vqa", "visual", "image", "vision"]
        configs = [cfg for cfg in configs if not any(kw in cfg.lower() for kw in vqa_keywords)]
        
        all_rows = []
        for cfg in configs:
            try:
                ds = load_dataset("madrylab/platinum-bench", cfg, split="test")
                ds = ds.filter(lambda x: x['platinum_target'] is not None)
                for row in ds:
                    all_rows.append(row)
            except Exception as e:
                print(f"   ⚠️ Skipping config '{cfg}': {e}")
                
        for row in tqdm(all_rows, desc="Filtering Platinum"):
            prompt = row.get("platinum_prompt_no_cot", "")
            target = row.get("platinum_target")
            if not prompt or not target or not isinstance(target, list) or len(target) == 0:
                continue
            response = target[0]
            if len(response.strip()) == 1 and response.strip().isalpha():
                response = response.strip().upper()
            processed.append({"prompt": prompt, "response": response})

    elif dataset_name == "mmlu":
        print("📂 Loading cais/mmlu...")
        configs = get_dataset_config_names("cais/mmlu")

        if n_samples and n_samples <= 2000:
            estimated_subjects_needed = min(len(configs), max(5, (n_samples // 100) + 2))
            configs = configs[:estimated_subjects_needed]
            print(f"   📌 Loading only {len(configs)} subjects for {n_samples} samples")

        for subj in tqdm(configs, desc="Processing MMLU Subjects"):
            if n_samples and len(processed) >= n_samples * 3:
                print(f"   ✂️ Early exit: collected {len(processed)} samples (target: {n_samples})")
                break

            try:
                test_ds = load_dataset("cais/mmlu", subj, split="test")

                # Pre-serialize dev examples once per subject for the few-shot prefix.
                dev_examples = []
                try:
                    dev_ds = load_dataset("cais/mmlu", subj, split="dev")
                except Exception:
                    try:
                        dev_ds = load_dataset("cais/mmlu", subj, split="validation")
                    except Exception:
                        dev_ds = []
                for d in dev_ds:
                    dev_examples.append({
                        "question": d["question"],
                        "choices": list(d["choices"]),
                        "answer": int(d["answer"]),
                    })

                # Store structured fields so build_example/format_sample can apply
                # the model's chat template at use-time.
                for row in test_ds:
                    processed.append({
                        "subject":      subj,
                        "question":     row["question"],
                        "choices":      list(row["choices"]),
                        "answer_idx":   int(row["answer"]),
                        "dev_examples": dev_examples,
                        "n_shot":       5,
                    })

            except Exception as e:
                print(f"   ⚠️ Skipping subject '{subj}': {e}")
            
    return processed


# Maps t1 split names → JSON filenames from generate_data_splits.py
_SPLIT_FILE_MAP = {
    "train_anchors": "train_anchors.json",
    "eval_anchors":  "eval_anchors.json",
    "pool":          "train_pool.json",
    "eval_pool":     "eval_pool.json",
    "warmup":        "warmup.json",
}

def load_from_json(data_dir: str, split: str, n_samples: int = None) -> list:
    """Load a pre-generated split from the JSON files produced by generate_data_splits.py."""
    filename = _SPLIT_FILE_MAP.get(split)
    if filename is None:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(_SPLIT_FILE_MAP.keys())}")
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Split file not found: {filepath}\nRun generate_data_splits.py first.")
    with open(filepath, 'r') as f:
        samples = json.load(f)
    if n_samples is not None:
        samples = samples[:n_samples]
    print(f"   ✓ Loaded {len(samples):,} '{split}' samples from {filepath}")
    return samples


def load_data_split(dataset_name: str, split: str, tokenizer, n_samples: int = None, anchor_size: int = 10000, pool_size: int = 100000):
    print(f"\n📂 Formatting {dataset_name} dataset for split: {split}")

    if dataset_name == "flan":
        if split == "eval_anchors":
            ds = load_dataset('sunweiwei/airrep-test', data_files='flan/test.jsonl', split='train')
            samples = list(ds)
            if n_samples is not None:
                samples = samples[:n_samples]
            
            processed = []
            for i, item in enumerate(tqdm(samples, desc="Formatting FLAN Eval")):
                prompt, response = format_item(item)
                processed.append({"doc_id": f"flan_{split}_{i}", "prompt": prompt, "response": response})
            return processed
        else:
            ds = list(load_dataset("Muennighoff/flan", split="train", verification_mode=VerificationMode.NO_CHECKS))
            np.random.seed(42)
            index = list(range(len(ds)))
            np.random.shuffle(index)

            if split == "train_anchors":      
                target_indices = index[:anchor_size]
            elif split == "pool":             
                target_indices = index[anchor_size : anchor_size + pool_size]
            elif split == "warmup":
                warmup_start = anchor_size + pool_size
                target_indices = index[warmup_start:] 
            else:
                raise ValueError(f"Unknown split: {split}")

            # Fallback for small datasets
            if not target_indices and len(index) > 0:
                target_indices = index[:min(len(index), n_samples or 1000)]

            if n_samples is not None:
                target_indices = target_indices[:n_samples]

            samples = [ds[i] for i in target_indices]
            
            processed = []
            for i, item in enumerate(tqdm(samples, desc=f"Formatting FLAN {split}")):
                prompt, response = format_item(item)
                processed.append({"doc_id": f"flan_{split}_{i}", "prompt": prompt, "response": response})
            
            print(f"   ✓ Loaded {len(processed):,} {split} samples")
            return processed
    else:
        # For Dolci, Platinum, and MMLU - PASS n_samples to avoid loading everything
        raw_samples = load_raw_dataset(dataset_name, tokenizer, n_samples=n_samples, split=split)
        
        np.random.seed(42)
        index = list(range(len(raw_samples)))
        np.random.shuffle(index)
        
        if split == "train_anchors":
            target_indices = index[:anchor_size]
        elif split == "pool":
            target_indices = index[anchor_size : anchor_size + pool_size]
        elif split == "warmup":
            start = anchor_size + pool_size
            target_indices = index[start : start + 100000]
        elif split == "eval_anchors":
            start = anchor_size + pool_size + 100000
            target_indices = index[start:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Fallback for small datasets
        if not target_indices and len(index) > 0:
            target_indices = index[:min(len(index), n_samples or 1000)]
            
        if n_samples is not None:
            target_indices = target_indices[:n_samples]
            
        if len(target_indices) < (n_samples or 0) and split != "warmup":
            print(f"   ⚠️ Warning: Only found {len(target_indices)} samples for split '{split}' (requested {n_samples})")
            
        processed = []
        for i, idx in enumerate(target_indices):
            item = raw_samples[idx]
            # Preserve every structured field (MMLU has question/choices/...,
            # FLAN/dolci/platinum have prompt/response). format_sample dispatches
            # on dataset_name and reads whichever fields it needs.
            processed.append({**item, "doc_id": f"{dataset_name}_{split}_{i}"})

        print(f"   ✓ Loaded {len(processed):,} {split} samples")
        return processed

# =========================================================================
# 3. LoRA Warmup -> Merge & Unload
# =========================================================================

class WarmupDataset(Dataset):
    def __init__(self, samples, tokenizer, dataset_name):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids, labels = build_example(sample, self.tokenizer, self.dataset_name)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
        }


def collate_fn(batch):
    valid_batch = [b for b in batch if not (b["labels"] == -100).all()]
    if not valid_batch:
        return None
        
    input_ids = pad_sequence(
        [b["input_ids"] for b in valid_batch], batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [b["labels"] for b in valid_batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "labels": labels}


def warmup_sft(model, tokenizer, pool_samples, n_warmup_samples: int, dataset_name: str, target_modules: list = None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print("\n" + "=" * 70)
    print("🔥 WARMUP STAGE: LoRA SFT -> Merge & Unload")
    print("=" * 70)

    train_samples = pool_samples[:n_warmup_samples]
    if not train_samples:
        print("   ⚠️ No warmup samples available. Skipping SFT stage.")
        return model

    print(f"   Using {len(train_samples)} samples for warmup")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = WarmupDataset(train_samples, tokenizer, dataset_name)
    dataloader = DataLoader(
        dataset,
        batch_size=WARMUP_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=WARMUP_LR)

    model.train()
    for epoch in range(WARMUP_EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Warmup Epoch {epoch+1}/{WARMUP_EPOCHS}")
        for batch in pbar:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print("\nMerging LoRA weights into Base Model...")
    model = model.merge_and_unload()

    print("Setting requires_grad for target module parameters...")
    for name, param in model.named_parameters():
        if "embed" in name or "lm_head" in name:
            param.requires_grad = False
        elif any(tm in name for tm in target_modules):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.eval()
    return model

# Test comment

# =========================================================================
# 4. Gradient Extraction & DB Storage
# =========================================================================

class GradientExtractor:
    def __init__(self, model, tokenizer, proj_dim, seeds, dataset_name, target_modules: list = None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.model = model
        self.tokenizer = tokenizer
        self.proj_dim = proj_dim
        self.seeds = seeds
        self.dataset_name = dataset_name
        self.target_modules = target_modules
        self.model.eval()

        self.target_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(tm in n for tm in target_modules)
        ]
        self.total_dim = sum(p.numel() for p in self.target_params)
        self.param_numels = [p.numel() for p in self.target_params]

        print(
            f"\n[Extractor] Target modules: {target_modules}"
        )
        print(
            f"[Extractor] Total gradient dim: {self.total_dim:,} "
            f"-> Projected to {proj_dim}"
        )

        # GRADIENT PRECISION FIX: Initialize natively in bfloat16 to avoid data loss
        self.buffer = torch.zeros(
            (GRAD_BATCH_SIZE, self.total_dim), dtype=torch.bfloat16, device=device
        )

    def _collect_batch(self, samples):
        for i, sample in enumerate(samples):
            self.model.zero_grad(set_to_none=True)

            input_ids, labels = build_example(sample, self.tokenizer, self.dataset_name)

            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            labels = torch.tensor(labels).unsqueeze(0).to(device)

            if (labels == -100).all():
                self.buffer[i].zero_()
                continue

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                loss = self.model(input_ids=input_ids, labels=labels).loss

            if torch.isnan(loss):
                self.buffer[i].zero_()
                continue

            grads = torch.autograd.grad(loss, self.target_params, allow_unused=True)

            offset = 0
            for g, numel in zip(grads, self.param_numels):
                if g is not None:
                    # GRADIENT PRECISION FIX: Direct copy in bfloat16, no premature casting
                    self.buffer[i, offset:offset + numel].copy_(g.reshape(-1))
                else:
                    self.buffer[i, offset:offset + numel].zero_()
                offset += numel

    @torch.no_grad()
    def _project_batch(self, raw_grads_bf16, seed):
        n = raw_grads_bf16.shape[0]
        D = self.total_dim
        d = self.proj_dim

        projected = torch.zeros((n, d), dtype=torch.float32, device=device)
        gen = torch.Generator(device=device).manual_seed(seed)
        chunk_size = 1_000_000

        for start in range(0, D, chunk_size):
            end = min(start + chunk_size, D)
            k = end - start

            hash_idx = torch.randint(
                0, d, (k,), generator=gen, device=device
            )
            signs = (
                torch.randint(0, 2, (k,), generator=gen, device=device,
                              dtype=torch.float32)
                .mul_(2)
                .sub_(1)
            )

            # GRADIENT PRECISION FIX: Cast to float32 locally before CountSketch arithmetic
            grad_chunk = raw_grads_bf16[:, start:end].float()
            signed = grad_chunk * signs
            projected.scatter_add_(
                1, hash_idx.unsqueeze(0).expand(n, -1), signed
            )

        return projected.cpu().numpy()

    def get_projected_gradients(self, samples):
        n = len(samples)
        results = {
            seed: np.zeros((n, self.proj_dim), dtype=np.float32)
            for seed in self.seeds
        }

        for i in range(0, n, GRAD_BATCH_SIZE):
            batch = samples[i : i + GRAD_BATCH_SIZE]
            b = len(batch)

            self.buffer[:b].zero_()
            self._collect_batch(batch)

            raw = self.buffer[:b]
            for seed in self.seeds:
                results[seed][i : i + b] = self._project_batch(raw, seed)

        return results

# =========================================================================
# SQLite Database Utilities
# =========================================================================

def init_db(db_path, dataset_name, split_name, proj_dim, seeds):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS projections (
            doc_id TEXT NOT NULL,
            projection_seed INTEGER NOT NULL,
            projected_gradient BLOB NOT NULL,
            PRIMARY KEY (doc_id, projection_seed)
        )
    """)
    cur.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
    )

    meta = {
        "dataset": dataset_name,
        "split": split_name,
        "proj_dim": str(proj_dim),
        "seeds": ",".join(str(s) for s in seeds),
        "warmup": "LoRA_MergeAndUnload",
    }
    for k, v in meta.items():
        cur.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (k, v))

    conn.commit()
    return conn, cur

# =========================================================================
# Main Execution
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stock gradients for Influence-Encoder training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", type=str, default="flan",
        choices=["flan", "dolci", "platinum", "mmlu"],
        help="Dataset to stock (flan, dolci, platinum, mmlu)"
    )
    parser.add_argument(
        "--split", type=str, required=True,
        choices=["pool", "train_anchors", "eval_anchors", "eval_pool"],
        help="Which split to stock."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to JSON splits produced by generate_data_splits.py. "
             "When set, loads data from disk instead of downloading from HuggingFace, "
             "guaranteeing the same partitioning as all downstream scripts."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Load and verify data splits only — skip model loading and gradient extraction."
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of samples to stock. If not specified, stocks all available samples for the split."
    )
    parser.add_argument(
        "--n_warmup", type=int, default=100000,
        help="Number of samples to use for LoRA SFT warmup (default: 100000)"
    )
    parser.add_argument(
        "--anchor_size", type=int, default=10000,
        help="The pool size reserved for train_anchors in the shuffle split (default: 10000)"
    )
    parser.add_argument(
        "--pool_size", type=int, default=100000,
        help="The pool size reserved for the candidate pool in the shuffle split (default: 100000)"
    )
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL,
        help=f"Model to use for gradient extraction (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--proj_dim", type=int, default=DEFAULT_PROJ_DIM,
        help=f"Projection dimension for CountSketch (default: {DEFAULT_PROJ_DIM})"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help=f"Random seeds for projections (default: {DEFAULT_SEEDS})"
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Quick test mode: limits warmup to 10 samples. Use with --n_samples for testing."
    )
    parser.add_argument(
        "--output_name", type=str, default=None,
        help="Custom output filename (without .sqlite extension)."
    )
    parser.add_argument(
        "--save_warmup_path", type=str, default=None,
        help="Path to save the model and tokenizer after the warmup SFT phase."
    )
    parser.add_argument(
        "--load_warmup_path", type=str, default=None,
        help="Path to load a pre-warmed model from, bypassing the warmup SFT phase."
    )
    parser.add_argument(
        "--target_modules", type=str, nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Module names to compute gradients for (default: LoRA target modules)"
    )
    parser.add_argument(
        "--force_recompute", action="store_true",
        help="If set, deletes existing database files and re-extracts all gradients. "
             "Default: False (resumes from existing DB)."
    )
    args = parser.parse_args()

    if args.output_name:
        db_path = f"{args.output_name}.sqlite"
    else:
        suffix = "_QUICK" if args.quick_test else ""
        db_path = f"{args.dataset}_{args.split}_d{args.proj_dim}_gradients{suffix}.sqlite"

    print("=" * 70)
    print("🧬 GRADIENT STOCKING PIPELINE")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Samples: {args.n_samples if args.n_samples else 'ALL AVAILABLE'}")
    print(f"Output: {db_path}")

    # ---- Data loading -------------------------------------------------------
    actual_n_warmup = 10 if args.quick_test else args.n_warmup

    if args.data_dir:
        print(f"\n📂 Loading splits from --data_dir: {args.data_dir}")
        warmup_samples = load_from_json(args.data_dir, "warmup", n_samples=actual_n_warmup)
        target_samples = load_from_json(args.data_dir, args.split, n_samples=args.n_samples)
    else:
        print("\n⚠️  No --data_dir provided; downloading from HuggingFace with legacy partitioning.")
        print("    This may produce DIFFERENT splits than generate_data_splits.py.")
        tokenizer_tmp = AutoTokenizer.from_pretrained(args.model_name)
        warmup_samples = load_data_split(
            args.dataset, "warmup", tokenizer_tmp,
            n_samples=actual_n_warmup,
            anchor_size=args.anchor_size, pool_size=args.pool_size
        )
        target_samples = load_data_split(
            args.dataset, args.split, tokenizer_tmp,
            n_samples=args.n_samples,
            anchor_size=args.anchor_size, pool_size=args.pool_size
        )

    # ---- Dry-run: verify data and exit without touching model ---------------
    if args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 DRY RUN — Data Verification (no model loaded)")
        print("=" * 70)
        print(f"  Warmup samples : {len(warmup_samples):,}")
        print(f"  Target ({args.split}) : {len(target_samples):,}")
        if warmup_samples:
            print(f"  Warmup doc_ids (first 3) : {[s['doc_id'] for s in warmup_samples[:3]]}")
        if target_samples:
            print(f"  Target doc_ids (first 3) : {[s['doc_id'] for s in target_samples[:3]]}")
        warmup_ids = {s['doc_id'] for s in warmup_samples}
        target_ids = {s['doc_id'] for s in target_samples}
        overlap = warmup_ids & target_ids
        print(f"  Warmup/target overlap    : {len(overlap)} (should be 0)")
        print("=" * 70)
        print("✅ Data verification complete — exiting (--dry_run).")
        return

    # ---- Model loading ------------------------------------------------------
    if args.load_warmup_path:
        print(f"\n🧠 Loading Pre-Warmed Model & Tokenizer: {args.load_warmup_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.load_warmup_path)
        ensure_chat_template(tokenizer)
            
        model = AutoModelForCausalLM.from_pretrained(
            args.load_warmup_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print("   Setting requires_grad for target module parameters...")
        for name, param in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                param.requires_grad = False
            elif any(tm in name for tm in args.target_modules):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print(f"\n🧠 Loading Model & Tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        ensure_chat_template(tokenizer)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        # Run the warmup phase
        model = warmup_sft(model, tokenizer, warmup_samples, actual_n_warmup, args.dataset, args.target_modules)
        
        if args.save_warmup_path:
            os.makedirs(args.save_warmup_path, exist_ok=True)
            print(f"\n💾 Saving Warmed-Up Model to {args.save_warmup_path}")
            model.save_pretrained(args.save_warmup_path)
            tokenizer.save_pretrained(args.save_warmup_path)

    print("\n" + "=" * 70)
    print(f"🎯 EXTRACTING GRADIENTS: {args.split}")
    print("=" * 70)

    extractor = GradientExtractor(model, tokenizer, args.proj_dim, args.seeds, args.dataset, args.target_modules)
    
    # Database initialization
    if args.force_recompute and os.path.exists(db_path):
        print(f"   🧹 --force_recompute set: removing existing database: {db_path}")
        os.remove(db_path)
    elif os.path.exists(db_path):
        print(f"   📂 Existing database found: {db_path} (resuming)")
        
    conn, cur = init_db(db_path, args.dataset, args.split, args.proj_dim, args.seeds)

    cur.execute("SELECT DISTINCT doc_id FROM projections")
    done_ids = set(row[0] for row in cur.fetchall())
    to_process = [s for s in target_samples if s["doc_id"] not in done_ids]

    if done_ids:
        print(f"   ℹ️  Skipping {len(done_ids)} already processed samples")
    
    print(f"   📊 Processing {len(to_process)} samples...")

    batch_data = []
    pbar = tqdm(total=len(to_process), desc="Stocking to SQLite")

    for item in to_process:
        batch_data.append(item)
        if len(batch_data) >= GRAD_BATCH_SIZE:
            seed_grads = extractor.get_projected_gradients(batch_data)

            doc_rows = []
            for d in batch_data:
                p_text, r_text = render_for_storage(d, args.dataset, tokenizer, MAX_SEQ_LEN)
                doc_rows.append((d["doc_id"], p_text, r_text))
            cur.executemany(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", doc_rows
            )
            proj_rows = []
            for seed in args.seeds:
                for i, d in enumerate(batch_data):
                    proj_rows.append(
                        (d["doc_id"], seed, seed_grads[seed][i].tobytes())
                    )
            cur.executemany(
                "INSERT OR REPLACE INTO projections VALUES (?, ?, ?)", proj_rows
            )
            conn.commit()

            pbar.update(len(batch_data))
            batch_data.clear()

    if batch_data:
        seed_grads = extractor.get_projected_gradients(batch_data)
        doc_rows = []
        for d in batch_data:
            p_text, r_text = render_for_storage(d, args.dataset, tokenizer, MAX_SEQ_LEN)
            doc_rows.append((d["doc_id"], p_text, r_text))
        cur.executemany(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", doc_rows
        )
        proj_rows = [
            (d["doc_id"], seed, seed_grads[seed][i].tobytes())
            for seed in args.seeds
            for i, d in enumerate(batch_data)
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO projections VALUES (?, ?, ?)", proj_rows
        )
        conn.commit()
        pbar.update(len(batch_data))

    pbar.close()
    conn.close()
    print("\n✅ STOCKING COMPLETE!")

if __name__ == "__main__":
    main()