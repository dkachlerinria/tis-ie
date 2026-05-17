"""Compute (num_anchors, num_train) score matrix from sentence-transformer
embeddings. Wraps the existing representation/embed pipeline so the artifact
contract matches the gradient methods.
"""

import argparse
import logging
import os
import time

import torch
from sentence_transformers import SentenceTransformer

from influence_eval.bbh_data import bbh_texts_for_encoder
from influence_eval.flops_measure import flop_counter
from influence_eval.model_utils import count_params  # only for the encoder param count
from representation.embed.compute_sentence_embeds import compute_train_embeddings
from representation.helper import batch_cosine_similarity


def _local_texts(path: str, n: int, start: int = 0) -> list:
    """Return n concatenated user+assistant strings from a local JSONL file, beginning at row `start`."""
    import json
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if len(texts) >= n:
                break
            item = json.loads(line)
            text = " ".join(m["content"] for m in item.get("messages", []))
            texts.append(text)
    return texts

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_model", required=True, type=str)
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_name", type=str, default="embedding")
    p.add_argument("--local_train_dataset", type=str, default=None,
                   help="Local JSONL path (e.g. dolly/dolly_data.jsonl). Uses first end_index rows for train and first num_anchors rows for anchors.")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Loading encoder: %s", args.encoder_model)
    model = SentenceTransformer(
        args.encoder_model, model_kwargs={"torch_dtype": torch.bfloat16}
    )
    if torch.cuda.is_available():
        model.to("cuda")
    tokenizer = model.tokenizer

    # Load anchor texts outside FlopCounterMode (no GPU work).
    if args.local_train_dataset:
        # Anchors are disjoint from train pool: rows [end_index : end_index + num_anchors].
        anchor_texts = _local_texts(args.local_train_dataset, args.num_anchors, start=args.end_index)
    else:
        anchor_texts = bbh_texts_for_encoder(n_samples=args.num_anchors, start_index=0)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        # Train embeddings (sliced to [0:end_index])
        train_embeds = compute_train_embeddings(
            model=model,
            tokenizer=tokenizer,
            train_dataset_path=args.local_train_dataset,  # None → tulu; path → local file
            batch_size=args.batch_size,
            start_index=0,
            end_index=args.end_index,
            debug=False,
        )
        # Anchor embeddings
        anchor_embeds = torch.from_numpy(
            model.encode(
                anchor_texts,
                batch_size=args.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        )
        # Score matrix
        scores = batch_cosine_similarity(
            dev_reps=anchor_embeds,
            train_reps=train_embeds,
            chunk_size=1024,
            normalize=False,
        ).float()
    inference_time_s = time.perf_counter() - t0
    measured_flops = int(counter.get_total_flops())
    n_samples = int(anchor_embeds.shape[0] + train_embeds.shape[0])
    time_per_sample_s = inference_time_s / max(n_samples, 1)
    logger.info(
        "Train embeds: %s, anchor embeds: %s, measured FLOPs: %.3e | Wall-clock: %.2fs (%.2fms/sample, batch_size=%d)",
        tuple(train_embeds.shape),
        tuple(anchor_embeds.shape),
        measured_flops,
        inference_time_s,
        1000 * time_per_sample_s,
        args.batch_size,
    )

    out_path = os.path.join(args.save_dir, f"{args.out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    params = count_params(model)
    meta = {
        **params,
        "model_name": args.encoder_model,
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "emb_dim": int(train_embeds.shape[1]),
        "measured_flops": measured_flops,
        "inference_time_s": float(inference_time_s),
        "time_per_sample_s": float(time_per_sample_s),
        "batch_size": int(args.batch_size),
    }
    torch.save(meta, os.path.join(args.save_dir, f"{args.out_name}_params.pt"))


if __name__ == "__main__":
    main()
