"""Stage-3 entry point: Train the AirRep encoder on (subset, dev, loss) triples.

Loads stage-1 pairs and stage-2 per-pair losses. For each pair we format:
  - subset texts: tis-ie chat-template strings (same format compute_train_embeddings
    uses at scoring time, so the encoder sees Tulu identically at train and inference).
  - dev texts: bbh_texts_for_encoder-style strings (same as Spearman-eval anchors).

GPU training is wrapped in `flop_counter()`; total written to `_flops_train.json`
under `--output_dir`. The encoder is then saved to that same dir as a standard
HuggingFace checkpoint loadable by `AirRep.from_pretrained(...)`.
"""

import argparse
import json
import logging
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from airrep.airrep_trainer import AirRepTrainer  # noqa: E402
from influence_eval.bbh_data import _ENCODER_PREFIX, load_bbh_samples  # noqa: E402
from influence_eval.flops_measure import flop_counter, save_phase_flops  # noqa: E402

logger = logging.getLogger(__name__)


MODES = {
    "tiny":   {"epochs": 1,  "batch_size": 1, "topk": 4,  "reference_size": 8,   "lr": 1e-4, "max_len": 256},
    "quick":  {"epochs": 3,  "batch_size": 1, "topk": 8,  "reference_size": 32,  "lr": 1e-4, "max_len": 384},
    "small":  {"epochs": 5,  "batch_size": 1, "topk": 16, "reference_size": 64,  "lr": 1e-4, "max_len": 384},
    "medium": {"epochs": 10, "batch_size": 1, "topk": 32, "reference_size": 128, "lr": 1e-4, "max_len": 512},
    "full":   {"epochs": 50, "batch_size": 1, "topk": 32, "reference_size": 256, "lr": 1e-4, "max_len": 512},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_path", required=True, type=str)
    p.add_argument("--loss_dir", required=True, type=str,
                   help="Directory containing loss-{pair_id}.json files from stage 2.")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--base_model", default="thenlper/gte-small", type=str)
    p.add_argument("--run_mode", default="tiny", choices=list(MODES.keys()))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--reference_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_len", type=int, default=None)
    return p.parse_args()


def _load_pairs(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_pairs_config(pairs_path: str):
    with open(os.path.join(os.path.dirname(pairs_path), "pairs_config.json")) as f:
        return json.load(f)


def _concat_messages_text(example: dict, tokenizer) -> str:
    """Same chat-template formatting as compute_train_embeddings (representation/embed)."""
    messages = example["messages"]
    eos = tokenizer.eos_token or ""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}{eos}\n")
        else:
            raise ValueError(f"unknown role {role}")
    text = "".join(parts)
    if tokenizer.bos_token:
        text = tokenizer.bos_token + text
    return text.strip()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = MODES[args.run_mode]
    epochs = args.epochs or cfg["epochs"]
    topk = args.topk or cfg["topk"]
    reference_size = args.reference_size or cfg["reference_size"]
    lr = args.lr if args.lr is not None else cfg["lr"]
    max_len = args.max_len or cfg["max_len"]

    os.makedirs(args.output_dir, exist_ok=True)
    flops_path = os.path.join(args.output_dir, "_flops_train.json")

    pairs = _load_pairs(args.pairs_path)
    pairs_cfg = _load_pairs_config(args.pairs_path)
    logger.info("Loaded %d pairs", len(pairs))

    # Load loss files in pair-list order.
    group_losses = []
    for pair in pairs:
        loss_path = os.path.join(args.loss_dir, f"loss-{pair['id']}.json")
        if not os.path.exists(loss_path):
            raise FileNotFoundError(f"Missing stage-2 output: {loss_path}")
        with open(loss_path) as f:
            data = json.load(f)
        if len(data["losses"]) != len(pair["dev"]):
            raise ValueError(
                f"Pair {pair['id']}: losses len {len(data['losses'])} != dev len {len(pair['dev'])}"
            )
        group_losses.append(data["losses"])
    group_losses_t = torch.tensor(group_losses, dtype=torch.float32)
    logger.info("Loaded group_losses tensor of shape %s", tuple(group_losses_t.shape))

    # ----- Format subset texts (Tulu chat-template).
    logger.info("Loading Tulu train pool and formatting subset texts...")
    tulu = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train")
    tulu = tulu.select(range(pairs_cfg["num_tulu"]))
    # Tokenizer just for bos/eos in template formatting; use base_model's tokenizer.
    tt = AutoTokenizer.from_pretrained(args.base_model)
    subset_texts = []
    for pair in pairs:
        subset_texts.append([_concat_messages_text(tulu[i], tt) for i in pair["select"]])

    # ----- Format dev texts (BBH, encoder-style prefix).
    logger.info("Loading BBH dev pool and formatting dev texts...")
    bbh_dev = load_bbh_samples(
        n_samples=pairs_cfg["dev_pool_size"],
        start_index=pairs_cfg["num_anchors"],
    )

    def _bbh_text(sample: dict) -> str:
        return f"{_ENCODER_PREFIX} {sample['prompt']} {sample['response']}".strip()

    dev_texts_by_id = {}
    for pair in pairs:
        if pair["dev_id"] in dev_texts_by_id:
            continue
        dev_texts_by_id[pair["dev_id"]] = [_bbh_text(bbh_dev[j]) for j in pair["dev"]]

    # ----- Train.
    trainer = AirRepTrainer(
        base_model=args.base_model,
        batch_size=1,
        epochs=epochs,
        lr=lr,
        topk=topk,
        reference_size=reference_size,
        max_len=max_len,
        save_path=args.output_dir,
    )

    with flop_counter() as counter:
        trainer.train(
            subset_texts=subset_texts,
            dev_texts_by_id=dev_texts_by_id,
            group_losses=group_losses_t,
            group_index=pairs,
        )
    training_flops = int(counter.get_total_flops())
    save_phase_flops(flops_path, training_flops)
    logger.info("AirRep encoder saved to %s (training FLOPs=%.3e)", args.output_dir, training_flops)


if __name__ == "__main__":
    main()
