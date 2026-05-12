"""Stage-2 SFT trainer for AirRep.

Fine-tunes a causal LM on a Tulu subset and evaluates per-example loss on a
BBH dev slice. Uses tis-ie's shared tokenization:
- `common.data.encode_with_messages_format` for Tulu chat-template subsets.
- A BBH-specific helper for dev samples (prompt + response with prompt mask).
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import encode_with_messages_format


def _encode_bbh_for_lm(sample: Dict, tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    """Tokenize a BBH `{prompt, response, ...}` sample into LM (input_ids, labels)
    with the prompt portion masked to -100. Mirrors `construct_test_sample` but
    uses the BBH key names directly.
    """
    prompt = sample["prompt"]
    response = sample["response"]

    prompt_ids = tokenizer(
        prompt, return_tensors="pt", max_length=max_length, truncation=True
    )["input_ids"][0]
    response_ids = tokenizer(
        response + tokenizer.eos_token,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"][0]

    input_ids = torch.cat([prompt_ids, response_ids], dim=0)[:max_length]
    labels = torch.cat(
        [torch.full_like(prompt_ids, -100), response_ids], dim=0
    )[:max_length]
    return {"input_ids": input_ids, "labels": labels}


class _TuluSubsetDataset(Dataset):
    def __init__(self, examples: List[Dict], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = encode_with_messages_format(ex, self.tokenizer, self.max_length)
        return enc["input_ids"], enc["labels"]


class _BBHDevDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = _encode_bbh_for_lm(self.samples[idx], self.tokenizer, self.max_length)
        return enc["input_ids"], enc["labels"]


def _collate(batch):
    input_ids, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class SFTTrainer:
    """Fine-tune a causal LM on a Tulu subset, then score per-example loss on BBH dev."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        epochs: int = 2,
        lr: float = 2e-5,
        max_length: int = 1024,
        use_flash_attn: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_length = max_length
        self.use_flash_attn = use_flash_attn
        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _new_model(self) -> AutoModelForCausalLM:
        kwargs = {"torch_dtype": torch.bfloat16}
        if self.use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        else:
            # eager for flop_counter compatibility with GQA models (see model_utils.py).
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        return model

    def train(self, train_examples: List[Dict]) -> AutoModelForCausalLM:
        model = self._new_model()
        dataset = _TuluSubsetDataset(train_examples, self.tokenizer, self.max_length)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate, num_workers=0
        )
        optimizer = AdamW(model.parameters(), lr=self.lr)
        model, optimizer, loader = self.accelerator.prepare(model, optimizer, loader)

        for epoch in range(self.epochs):
            self.accelerator.print(f"  SFT epoch {epoch + 1}/{self.epochs}")
            model.train()
            losses = []
            for batch in tqdm(loader, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(model):
                    out = model(**batch)
                    loss = out.loss
                    if torch.isnan(loss):
                        loss = torch.tensor(0.0, requires_grad=True, device=loss.device)
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
            if losses:
                self.accelerator.print(f"  SFT epoch {epoch + 1} avg loss: {sum(losses) / len(losses):.4f}")
        return self.accelerator.unwrap_model(model)

    @torch.no_grad()
    def evaluate(self, model: AutoModelForCausalLM, dev_samples: List[Dict]) -> List[float]:
        model.eval()
        dataset = _BBHDevDataset(dev_samples, self.tokenizer, self.max_length)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=_collate, num_workers=0
        )
        device = next(model.parameters()).device
        out_losses: List[float] = []
        for batch in tqdm(loader, disable=not self.accelerator.is_main_process):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            bs = logits.size(0)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(bs, -1)
            seq_len = batch["labels"][..., 1:].ne(-100).sum(dim=-1).clamp(min=1)
            per_example = loss.sum(dim=-1) / seq_len
            no_target = batch["labels"][..., 1:].ne(-100).sum(dim=-1) == 0
            per_example = torch.where(
                no_target, torch.full_like(per_example, 10.0), per_example
            )
            out_losses.extend(per_example.float().cpu().tolist())
        return out_losses
