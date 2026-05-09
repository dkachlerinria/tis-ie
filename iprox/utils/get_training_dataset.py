import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset
import logging
import sys

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0):
    """ get training dataset with a specified seed """

    raw_datasets = load_raw_dataset(
        train_files, sample_percentage=sample_percentage, seed=seed)
    lm_datasets = encode_data(
        raw_datasets, tokenizer, max_seq_length)
    return lm_datasets


def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset(
        "json",
        data_files=train_files,
    )["train"]
    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    with temp_seed(seed):
        index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset

def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=10, overwrite_cache=False):
    """ encode data with the specified tokenizer and the chat format. """
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length)
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    return lm_datasets

def get_encode_function(raw_datasets, tokenizer, max_seq_length):
    """ get encode function based on the dataset. """
    return partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

def encode_with_messages_format(example, tokenizer, max_seq_length):
    """Encode a sample using the unified format_instruction.

    Reduces all callers (single-turn FLAN/dolci/platinum, or messages-format
    items with one user → one assistant turn) to the same chat-templated
    rendering used by gradient_stocking, encoder training, SFT training, and
    sacred eval. Returns torch tensors for HF datasets.map() compatibility.
    """
    import os, sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from formatting import format_instruction, ensure_chat_template
    import torch

    ensure_chat_template(tokenizer)

    # Normalize input → (prompt, response).
    messages = example.get('messages')
    if messages:
        # Take last assistant turn as response, everything before its first
        # appearance as the user/system prelude collapsed into a single user
        # turn. For our actual pipeline (single-turn data) this is equivalent
        # to {prompt, response}.
        last_ass_idx = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if messages[i].get("role") == "assistant"
             and messages[i].get("content")
             and not messages[i]["content"].isspace()),
            None,
        )
        if last_ass_idx is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        response = messages[last_ass_idx]["content"]
        # Concatenate prior turns (user/system) into the user-content prompt.
        prior = [m for m in messages[:last_ass_idx] if m.get("role") in ("user", "system")]
        prompt = "\n\n".join(m["content"] for m in prior if m.get("content")) or ""
    else:
        prompt = example.get('prompt') or ''
        response = example.get('completion') or example.get('response') or ''

    if not prompt or not response or response.isspace():
        return {"input_ids": [], "attention_mask": [], "labels": []}

    out = format_instruction(prompt, response, tokenizer, max_seq_length)
    return {
        "input_ids":      torch.tensor(out["input_ids"]),
        "attention_mask": torch.tensor(out["attention_mask"]),
        "labels":         torch.tensor(out["labels"]),
    }