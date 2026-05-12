"""Model loading helpers for the influence-spearman experiment.

The key entrypoint is `load_base_with_fresh_lora`: load any HF causal LM
and wrap it with a deterministic LoRA adapter on the configured target
modules. This is what lets us compute "LoRA-only gradients on the base
model" without any warmup training.

The fresh-LoRA construction is the single point that knows about model
swapping: caller passes `model_name`, we handle the rest. `all-linear`
target works for any standard transformer (Qwen, Llama, etc.).
"""

from typing import Any, List, Union

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_target_modules(target_modules: str) -> Union[str, List[str]]:
    if target_modules == "all-linear":
        return "all-linear"
    return [t.strip() for t in target_modules.split(",") if t.strip()]


def load_base_with_fresh_lora(
    model_name: str,
    tokenizer: AutoTokenizer,
    lora_target_modules: str = "all-linear",
    lora_rank: int = 128,
    lora_alpha: int = 512,
    lora_dropout: float = 0.1,
    seed: int = 0,
    torch_dtype: Any = torch.bfloat16,
) -> PeftModel:
    # attn_implementation="sdpa" pinned for FLOP-measurement reproducibility.
    # See KNOWN_ISSUES.txt for rationale and fallback if SDPA breaks.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, attn_implementation="sdpa"
    )
    base_model.to("cuda")

    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        print(f"Resizing embeddings: {embedding_size} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))

    torch.manual_seed(seed)
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=_parse_target_modules(lora_target_modules),
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Fresh LoRA attached: {trainable:,} trainable / {total:,} total params "
        f"({100 * trainable / total:.2f}%)"
    )

    return model


def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
