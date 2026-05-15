"""Compute (num_anchors, num_train) score matrix from a trained AirRep encoder.

Loads the encoder from --encoder_dir (output of airrep.train_airrep_encoder),
embeds the standard Spearman-eval BBH anchors and Tulu train pool using the
same data loading utilities as compute_embedding_scores.py / compute_influcoder_scores.py,
and writes the score matrix + params dict.

FLOPs accounting matches influcoder's multi-phase pattern:
  measured_flops = sft_flops (stage 2) + training_flops (stage 3) + inference_flops (this script)
"""

import argparse
import logging
import os
import time

import torch

from influence_eval.bbh_data import bbh_texts_for_encoder
from influence_eval.flops_measure import flop_counter, load_phase_flops, load_phase_timing
from influence_eval.model_utils import count_params
from representation.embed.compute_sentence_embeds import compute_train_embeddings
from representation.helper import batch_cosine_similarity

# Adding the repo root to sys.path is not needed here — this file lives inside
# influence_eval/, which is on the path when invoked via `python -m`.
from airrep.modeling_airrep import AirRep

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_dir", required=True, type=str)
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_name", type=str, default="airrep")
    p.add_argument("--sft_flops_path", type=str, default=None,
                   help="Path to _flops_sft.json (stage 2). Defaults to 0 if missing.")
    p.add_argument("--training_flops_path", type=str, default=None,
                   help="Path to _flops_train.json (stage 3). Defaults to 0 if missing.")
    p.add_argument("--sft_timing_path", type=str, default=None,
                   help="Path to _timing_sft.json (stage 2). Defaults to 0 if missing.")
    p.add_argument("--training_timing_path", type=str, default=None,
                   help="Path to _timing_train.json (stage 3). Defaults to 0 if missing.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Loading trained AirRep encoder from: %s", args.encoder_dir)
    airrep = AirRep.from_pretrained(args.encoder_dir)

    # Anchor texts (BBH [0:num_anchors]) — outside flop_counter (no GPU work).
    anchor_texts = bbh_texts_for_encoder(n_samples=args.num_anchors, start_index=0)

    t0 = time.perf_counter()
    with flop_counter() as counter:
        train_embeds = compute_train_embeddings(
            model=airrep,  # AirRep.encode is SentenceTransformer-compatible
            tokenizer=airrep.tokenizer,
            train_dataset_path=None,
            batch_size=args.batch_size,
            start_index=0,
            end_index=args.end_index,
            debug=False,
        )
        anchor_embeds_np = airrep.encode(
            anchor_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        anchor_embeds = torch.from_numpy(anchor_embeds_np)

        scores = batch_cosine_similarity(
            dev_reps=anchor_embeds,
            train_reps=train_embeds,
            chunk_size=1024,
            normalize=False,
        ).float()
    inference_time_s = time.perf_counter() - t0
    inference_flops = int(counter.get_total_flops())
    n_samples = int(anchor_embeds.shape[0] + train_embeds.shape[0])
    inference_time_per_sample_s = inference_time_s / max(n_samples, 1)

    sft_flops = load_phase_flops(args.sft_flops_path) if args.sft_flops_path else 0
    training_flops = load_phase_flops(args.training_flops_path) if args.training_flops_path else 0
    total_flops = sft_flops + training_flops + inference_flops
    sft_time_s = load_phase_timing(args.sft_timing_path) if args.sft_timing_path else 0.0
    training_time_s = load_phase_timing(args.training_timing_path) if args.training_timing_path else 0.0
    total_time_s = sft_time_s + training_time_s + inference_time_s

    logger.info(
        "FLOPs — sft=%.3e, training=%.3e, inference=%.3e, total=%.3e",
        sft_flops, training_flops, inference_flops, total_flops,
    )
    logger.info(
        "Wall-clock — inference=%.2fs (%.2fms/sample), total=%.2fs",
        inference_time_s, 1000 * inference_time_per_sample_s, total_time_s,
    )
    if sft_flops == 0:
        logger.warning("No SFT FLOPs at %s — was airrep.sft_subsets re-run after instrumentation?",
                       args.sft_flops_path)
    if training_flops == 0:
        logger.warning("No training FLOPs at %s — was airrep.train_airrep_encoder re-run after instrumentation?",
                       args.training_flops_path)

    out_path = os.path.join(args.save_dir, f"{args.out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    encoder_params = count_params(airrep.model)
    meta = {
        **encoder_params,
        "emb_dim": int(train_embeds.shape[1]),
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "model_name": args.encoder_dir,
        "sft_flops": int(sft_flops),
        "training_flops": int(training_flops),
        "inference_flops": int(inference_flops),
        "measured_flops": int(total_flops),
        "inference_time_s": float(inference_time_s),
        "time_per_sample_s": float(inference_time_per_sample_s),
        "measured_time_s": float(total_time_s),
    }
    torch.save(meta, os.path.join(args.save_dir, f"{args.out_name}_params.pt"))
    logger.info("Saved params for FLOPS accounting")


if __name__ == "__main__":
    main()
