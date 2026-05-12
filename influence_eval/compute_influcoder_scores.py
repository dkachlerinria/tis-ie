"""Compute (num_anchors, num_train) score matrix using a trained influence encoder.

Loads the encoder from --encoder_dir (output of train_influence_encoder.py),
then embeds the standard Spearman-eval anchors and train pool using the same
data loading functions as compute_embedding_scores.py.
"""

import argparse
import logging
import os

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM

from influence_eval.bbh_data import bbh_texts_for_encoder
from influence_eval.flops_measure import flop_counter, load_phase_flops
from influence_eval.model_utils import count_params
from representation.embed.compute_sentence_embeds import compute_train_embeddings
from representation.helper import batch_cosine_similarity

logger = logging.getLogger(__name__)


def _count_gradient_model_params(model_name: str) -> dict:
    """Load gradient model on CPU just to count parameters, then discard."""
    logger.info("Loading gradient model on CPU to count params: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    total = sum(p.numel() for p in model.parameters())
    linear = sum(
        m.weight.numel()
        for m in model.modules()
        if isinstance(m, torch.nn.Linear)
    )
    del model
    return {"grad_model_total": total, "grad_model_linear": linear}


def compute_influcoder_scores(
    encoder_dir: str,
    gradient_model: str,
    save_dir: str,
    end_index: int,
    num_anchors: int,
    dev_dataset_name: str,
    n_stock_anchors: int,
    n_stock_pool: int,
    proj_dim: int,
    stocking_flops_path: str,
    training_flops_path: str,
    batch_size: int = 32,
    out_name: str = "influcoder",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    logger.info("Loading trained influence encoder from: %s", encoder_dir)
    model = SentenceTransformer(encoder_dir)
    if torch.cuda.is_available():
        model.to("cuda")
    tokenizer = model.tokenizer

    anchor_texts = bbh_texts_for_encoder(n_samples=num_anchors, start_index=0)

    with flop_counter() as counter:
        train_embeds = compute_train_embeddings(
            model=model,
            tokenizer=tokenizer,
            train_dataset_path=None,
            batch_size=batch_size,
            start_index=0,
            end_index=end_index,
            debug=False,
        )
        anchor_embeds = torch.from_numpy(
            model.encode(
                anchor_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        )
        scores = batch_cosine_similarity(
            dev_reps=anchor_embeds,
            train_reps=train_embeds,
            chunk_size=1024,
            normalize=False,
        ).float()
    inference_flops = int(counter.get_total_flops())

    stocking_flops = load_phase_flops(stocking_flops_path)
    training_flops = load_phase_flops(training_flops_path)
    total_flops = stocking_flops + training_flops + inference_flops
    logger.info(
        "FLOPs — stocking=%.3e, training=%.3e, inference=%.3e, total=%.3e",
        stocking_flops, training_flops, inference_flops, total_flops,
    )
    if stocking_flops == 0:
        logger.warning("No stocking FLOPs at %s — was gradient_stocking re-run after instrumentation?",
                       stocking_flops_path)
    if training_flops == 0:
        logger.warning("No training FLOPs at %s — was train_influence_encoder re-run after instrumentation?",
                       training_flops_path)

    out_path = os.path.join(save_dir, f"{out_name}_scores.pt")
    torch.save(scores, out_path)
    logger.info("Saved score matrix: %s shape=%s", out_path, tuple(scores.shape))

    encoder_params = count_params(model)
    grad_params = _count_gradient_model_params(gradient_model)

    meta = {
        **encoder_params,
        **grad_params,
        "emb_dim": int(train_embeds.shape[1]),
        "num_anchors": int(scores.shape[0]),
        "num_train": int(scores.shape[1]),
        "model_name": encoder_dir,
        "n_stock_anchors": int(n_stock_anchors),
        "n_stock_pool": int(n_stock_pool),
        "proj_dim": int(proj_dim),
        "stocking_flops": int(stocking_flops),
        "training_flops": int(training_flops),
        "inference_flops": int(inference_flops),
        "measured_flops": int(total_flops),
    }
    torch.save(meta, os.path.join(save_dir, f"{out_name}_params.pt"))
    logger.info("Saved params for FLOPS accounting")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_dir", required=True, type=str)
    p.add_argument("--gradient_model", required=True, type=str,
                   help="HF model used for gradient stocking (for FLOPS param count).")
    p.add_argument("--save_dir", required=True, type=str)
    p.add_argument("--end_index", type=int, required=True)
    p.add_argument("--num_anchors", type=int, required=True)
    p.add_argument("--dev_dataset_name", type=str, default="bbh")
    p.add_argument("--n_stock_anchors", type=int, required=True,
                   help="N_TRAIN_A + N_EVAL_A (for FLOPS).")
    p.add_argument("--n_stock_pool", type=int, required=True,
                   help="N_TRAIN_P + N_EVAL_P (for FLOPS).")
    p.add_argument("--proj_dim", type=int, required=True,
                   help="INFLUCODER_PROJ_DIM (for FLOPS).")
    p.add_argument("--stocking_flops_path", required=True, type=str,
                   help="${INFLUCODER_DB_DIR}/_flops.json — written by gradient_stocking.py.")
    p.add_argument("--training_flops_path", required=True, type=str,
                   help="${INFLUCODER_ENCODER_DIR}/_flops.json — written by train_influence_encoder.py.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_name", type=str, default="influcoder")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    compute_influcoder_scores(
        encoder_dir=args.encoder_dir,
        gradient_model=args.gradient_model,
        save_dir=args.save_dir,
        end_index=args.end_index,
        num_anchors=args.num_anchors,
        dev_dataset_name=args.dev_dataset_name,
        n_stock_anchors=args.n_stock_anchors,
        n_stock_pool=args.n_stock_pool,
        proj_dim=args.proj_dim,
        stocking_flops_path=args.stocking_flops_path,
        training_flops_path=args.training_flops_path,
        batch_size=args.batch_size,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
