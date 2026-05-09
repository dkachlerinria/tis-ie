"""Unified text formatting for the influence-encoder pipeline.

ONE format per dataset, applied consistently across:
  - gradient stocking (gradient_stocking.py)
  - encoder training (train_influence_encoder.py)
  - encoder selection at inference time (data_selection_sft_benchmark.py)
  - SFT training (data_selection_sft_benchmark.py SFTDataset, warmup_model.py)
  - LoGra/IProX selection-time gradients
  - sacred MMLU/BBH eval (run_sacred_mmlu_eval)
  - lm-eval benchmark (run_mmlu_eval — passes apply_chat_template=True)

All formatting goes through tokenizer.apply_chat_template using the model's
native chat template. For base models without one, ChatML is set as a default
by ensure_chat_template().

Public API:
  ensure_chat_template(tokenizer)
  format_instruction(prompt, response, tokenizer, max_seq_len)
  format_bbh_eval(question, answer, tokenizer, fewshot_text, ...)
  format_bbh_eval(question, answer, tokenizer, fewshot_text, ...)
  format_sample(item, dataset_name, tokenizer, max_seq_len)   # dispatcher
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional


# ChatML is the fallback for base models without a configured template.
# Qwen2.5/Qwen3 vocab includes <|im_start|> and <|im_end|> as single tokens.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


def ensure_chat_template(tokenizer):
    """Set ChatML as a fallback if the tokenizer has no chat_template.

    Also fills in pad_token from eos_token if missing, since base-model
    tokenizers commonly leave pad_token unset.
    """
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = CHATML_TEMPLATE
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _flat_ids(x) -> List[int]:
    """Flatten apply_chat_template / tokenizer output to a list of ints."""
    if hasattr(x, "input_ids"):
        return _flat_ids(x.input_ids)
    if hasattr(x, "get") and "input_ids" in x:
        return _flat_ids(x["input_ids"])
    if hasattr(x, "tolist"):
        return _flat_ids(x.tolist())
    if isinstance(x, (list, tuple)):
        if not x:
            return []
        if isinstance(x[0], (list, tuple)) or hasattr(x[0], "input_ids"):
            return _flat_ids(x[0])
        return [int(v) for v in x]
    return [int(x)]


# ---------------------------------------------------------------------------
# Single-turn instruction formatter (FLAN, dolci, platinum, ...)
# ---------------------------------------------------------------------------

def format_instruction(
    prompt: str,
    response: str,
    tokenizer,
    max_seq_len: int = 1024,
) -> Dict[str, Any]:
    """Render {prompt, response} as a 2-turn chat conversation.

    Returns:
        dict with:
          input_ids       : List[int]   — full chat-templated tokenization
          labels          : List[int]   — response-only (prompt masked to -100)
          attention_mask  : List[int]
          full_text       : str         — chat-templated string with both turns
          prompt_text     : str         — chat-templated string ending at assistant
                                          generation prompt (no response yet)
    """
    ensure_chat_template(tokenizer)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = _flat_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    ))

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = _flat_ids(tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    ))

    input_ids = full_ids[:max_seq_len]
    p_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * p_len + input_ids[p_len:]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "full_text": full_text,
        "prompt_text": prompt_text,
    }


# ---------------------------------------------------------------------------
# BBH formatter (3-shot CoT). Caller supplies the standard task prompt + demos.
# ---------------------------------------------------------------------------

def format_bbh_eval(
    question: str,
    answer: str,
    tokenizer,
    fewshot_text: str = "",
    max_seq_len: int = 2048,
    answer_cue: str = "A:",
) -> Dict[str, Any]:
    """Render a BBH item with chat template; cue ends at "A:" for continuation.

    fewshot_text is the canonical BBH task prompt with demos, *not* including
    the trailing "A:" of the test item.
    """
    ensure_chat_template(tokenizer)

    if fewshot_text.strip():
        user_content = fewshot_text.strip() + "\n\nQ: " + question.strip()
    else:
        user_content = "Q: " + question.strip()

    assistant_content = f"{answer_cue} {answer}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = _flat_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    ))

    gen_prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    eval_prompt_text = gen_prefix + answer_cue

    input_ids = full_ids[:max_seq_len]
    eval_prefix_ids = _flat_ids(
        tokenizer(eval_prompt_text, add_special_tokens=False)
    )
    p_len = min(len(eval_prefix_ids), len(input_ids))
    labels = [-100] * p_len + input_ids[p_len:]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "full_text": full_text,
        "eval_prompt_text": eval_prompt_text,
        "response_text": answer,
    }


# ---------------------------------------------------------------------------
# Per-dataset dispatcher. Use this everywhere in the pipeline.
# ---------------------------------------------------------------------------

def format_sample(
    item: Dict[str, Any],
    dataset_name: str,
    tokenizer,
    max_seq_len: int = 1024,
) -> Dict[str, Any]:
    """Dispatch to the correct formatter for the given dataset.

    A single dataset_name → exactly one format. No exceptions.
    """
    name = (dataset_name or "").lower()
    if name in ("flan", "dolci", "platinum", "instruction", "dolly"):
        return format_instruction(
            item["prompt"], item["response"], tokenizer, max_seq_len
        )
    if name == "bbh":
        return format_bbh_eval(
            question=item["question"],
            answer=item["answer"],
            tokenizer=tokenizer,
            fewshot_text=item.get("fewshot_text", ""),
            max_seq_len=max_seq_len,
        )
    # For MMLU (and any other dataset): items always have {prompt, response}
    # strings rendered by lm-eval at generation time — treat as instruction.
    return format_instruction(
        item["prompt"], item["response"], tokenizer, max_seq_len
    )


# ---------------------------------------------------------------------------
# Convenience tuple-returning shims for hot code paths that just want
# (input_ids, labels). Strictly equivalent to format_instruction(...)["..."].
# ---------------------------------------------------------------------------

def encode_instruction(prompt: str, response: str, tokenizer, max_seq_len: int = 1024):
    """Shim: returns (input_ids, labels) tuple for callers that don't need full dict."""
    out = format_instruction(prompt, response, tokenizer, max_seq_len)
    return out["input_ids"], out["labels"]


def render_instruction_text(prompt: str, response: str, tokenizer) -> str:
    """Shim: returns just the chat-rendered text string."""
    return format_instruction(prompt, response, tokenizer, max_seq_len=10**9)["full_text"]



def render_for_storage(
    item: Dict[str, Any],
    dataset_name: str,
    tokenizer,
    max_seq_len: int = 1024,
):
    """Render chat-templated text and split into (prompt_part, response_part).

    Used by gradient_stocking to populate the (prompt, response) columns of
    its SQLite DB. Concatenating the two parts yields the full chat-templated
    rendering — the same text the gradient was computed against. Downstream
    consumers (e.g. samples_to_texts) just do prompt + response to recover it.
    """
    fmt = format_sample(item, dataset_name, tokenizer, max_seq_len)
    full_text = fmt["full_text"]
    prefix = fmt.get("prompt_text") or fmt.get("eval_prompt_text") or ""
    if prefix and full_text.startswith(prefix):
        return prefix, full_text[len(prefix):]
    # Fallback: chat template doesn't produce a strict prefix; stash all in
    # the prompt column. This shouldn't trigger for ChatML or any well-formed
    # template, but keeps behavior defined.
    return full_text, ""
