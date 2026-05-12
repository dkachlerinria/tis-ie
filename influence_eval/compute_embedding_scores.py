"""Compute (num_anchors, num_train) score matrix from sentence-transformer
embeddings. Wraps the existing representation/embed pipeline so the artifact
contract matches the gradient methods.
"""

import argparse
import logging
import os

import torch
from sentence_transformers import SentenceTransformer

from influence_eval.bbh_data import bbh_texts_for_encoder
from influence_eval.flops_measure import flop_counter
from influence_eval.model_utils import count_params  # only for the encoder param count
from representation.embed.compute_sentence_embeds import compute_train_embeddings
from representation.helper import batch_cosine_similarity

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
    anchor_texts = bbh_texts_for_encoder(n_samples=args.num_anchors, start_index=0)

    with flop_counter() as counter:
        # Train embeddings (sliced to [0:end_index])
        train_embeds = compute_train_embeddings(
            model=model,
            tokenizer=tokenizer,
            train_dataset_path=None,
            batch_size=args.batch_size,
            start_index=0,
            end_index=args.end_index,
            debug=False,
        )
        # Anchor embeddings from local BBH [0:num_anchors] (same slice as ground truth)
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
    measured_flops = int(counter.get_total_flops())
    logger.info(
        "Train embeds: %s, anchor embeds: %s, measured FLOPs: %.3e",
        tuple(train_embeds.shape),
        tuple(anchor_embeds.shape),
        measured_flops,
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
    }
    torch.save(meta, os.path.join(args.save_dir, f"{args.out_name}_params.pt"))


if __name__ == "__main__":
    main()
