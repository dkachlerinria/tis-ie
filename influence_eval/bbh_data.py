"""Load local BBH data for the influence-Spearman eval pipeline.

Uses the same fixed seed=42 shuffle as influcoder/gradient_stocking.py so
that index i maps to the same BBH example across all pipeline steps.

Spearman eval anchors: [0 : NUM_ANCHORS]
Influcoder train_anchors: [NUM_ANCHORS : NUM_ANCHORS + N_TRAIN_A]
Influcoder eval_anchors:  [NUM_ANCHORS + N_TRAIN_A : ...]
"""

import glob
import json
import os
from typing import List, Optional

import numpy as np


def _find_bbh_dir() -> str:
    for d in ("data/eval/bbh", "data/bbh"):
        if os.path.exists(d):
            return d
    raise FileNotFoundError(
        "BBH data not found. Run download_eval.sh first (expects data/eval/bbh)."
    )


def load_bbh_samples(
    n_samples: Optional[int] = None,
    start_index: int = 0,
    bbh_dir: Optional[str] = None,
) -> List[dict]:
    """Return list of {"prompt", "response"} dicts shuffled with seed=42."""
    if bbh_dir is None:
        bbh_dir = _find_bbh_dir()

    bbh_tasks_dir = os.path.join(bbh_dir, "bbh")
    prompt_dir = os.path.join(bbh_dir, "cot-prompts")
    if not os.path.exists(bbh_tasks_dir) or not os.path.exists(prompt_dir):
        raise FileNotFoundError(
            f"BBH subdirs not found: {bbh_tasks_dir}, {prompt_dir}"
        )

    all_tasks = {}
    for task_file in glob.glob(os.path.join(bbh_tasks_dir, "*.json")):
        task_name = os.path.basename(task_file).split(".")[0]
        with open(task_file) as f:
            all_tasks[task_name] = json.load(f)["examples"]

    all_prompts = {}
    for cot_file in glob.glob(os.path.join(prompt_dir, "*.txt")):
        task_name = os.path.basename(cot_file).split(".")[0]
        with open(cot_file) as f:
            all_prompts[task_name] = "".join(f.readlines()[2:])

    processed = []
    for task_name, examples in all_tasks.items():
        task_prompt = all_prompts.get(task_name, "").strip()
        for ex in examples:
            processed.append({
                "prompt": task_prompt + "\n\nQ: " + ex["input"],
                "response": ex["target"],
                "task": task_name,
            })

    # Same shuffle as gradient_stocking.py for consistent indexing
    np.random.seed(42)
    indices = np.arange(len(processed))
    np.random.shuffle(indices)
    processed = [processed[i] for i in indices]

    end = start_index + n_samples if n_samples is not None else len(processed)
    return processed[start_index:end]


_ENCODER_PREFIX = (
    "Instruct: Given a sample, find the passages closest to that sample.\nQuery:"
)


def bbh_texts_for_encoder(
    n_samples: Optional[int] = None,
    start_index: int = 0,
) -> List[str]:
    """Return formatted text strings for SentenceTransformer encoding."""
    samples = load_bbh_samples(n_samples=n_samples, start_index=start_index)
    return [f"{_ENCODER_PREFIX} {s['prompt']} {s['response']}".strip() for s in samples]
