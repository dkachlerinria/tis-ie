"""Stage-3: Train AirRep encoder on (subset-text, dev-text, loss) tuples.

Adapted from AirRep-main/airrep/airrep_trainer.py for tis-ie. Single-GPU
implementation kept simple; the distributed broadcast/gather logic from
upstream is dropped since tis-ie's other encoder trainers (e.g.
influcoder/train_influence_encoder.py) are also single-GPU. The core RankNet
loss + attention-pooled subset/dev cosine similarity is preserved.

Inputs:
- `subset_texts`: list[list[str]] — for each pair, the pre-formatted subset texts.
- `dev_texts`: dict[dev_id -> list[str]] — per dev split, the pre-formatted dev texts.
- `group_losses`: torch.Tensor (n_pairs, dev_size) — per-example losses from stage 2.
- `group_index`: list[dict] — pair metadata (id, select, dev, dev_id).
"""

from collections import defaultdict
from itertools import product
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.optimization import get_constant_schedule_with_warmup

from .modeling_airrep import AirRepConfig, AirRepModel


def ranknet_loss(y_pred: torch.Tensor, y_true: torch.Tensor, weight_by_diff: bool = True, clip_median: bool = True) -> torch.Tensor:
    pairs = list(product(range(y_true.shape[1]), repeat=2))
    pairs_true = y_true[:, pairs]
    selected_pred = y_pred[:, pairs]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
    pred_diffs = pred_diffs[mask]

    weight = torch.abs(true_diffs)[mask] if weight_by_diff else None
    if weight is not None and clip_median:
        weight = torch.clamp(weight, min=0.0, max=10.0)

    targets = (true_diffs > 0).float()[mask]
    if pred_diffs.numel() == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    return BCEWithLogitsLoss(weight=weight)(pred_diffs, targets)


def eval_lds(scores: np.ndarray, labels: np.ndarray) -> float:
    out = []
    for i in range(len(scores)):
        c, _ = stats.spearmanr(scores[i], labels[i])
        out.append(0.0 if np.isnan(c) else float(c))
    return float(np.mean(out)) if out else 0.0


class AirRepDataset(Dataset):
    """Yields ([subset_input_ids]*topk, [dev_input_ids]*reference_size, target (reference_size, topk))."""

    def __init__(
        self,
        subset_texts: List[List[str]],
        dev_texts_by_id: Dict[int, List[str]],
        tokenizer,
        group_losses: torch.Tensor,
        group_index: List[Dict],
        topk: int = 32,
        reference_size: int = 1000,
        max_len: int = 512,
    ):
        if group_losses.shape[0] != len(group_index):
            raise ValueError(
                f"group_losses rows={group_losses.shape[0]} != len(group_index)={len(group_index)}"
            )
        if len(subset_texts) != len(group_index):
            raise ValueError(
                f"len(subset_texts)={len(subset_texts)} != len(group_index)={len(group_index)}"
            )

        self.subset_texts = subset_texts
        self.dev_texts_by_id = dev_texts_by_id
        self.tokenizer = tokenizer
        self.group_index = group_index
        self.topk = topk
        self.reference_size = reference_size
        self.max_len = max_len

        # Attach loss to each pair and group by dev_id.
        self.batched: Dict[int, List[Dict]] = defaultdict(list)
        for i, item in enumerate(group_index):
            item = dict(item)
            item["loss"] = group_losses[i].tolist()
            item["_pair_idx"] = i
            self.batched[item["dev_id"]].append(item)

        # Per-dev-id baseline (mean/std) for loss z-scoring.
        baseline_collect: Dict[int, List[float]] = defaultdict(list)
        for i, item in enumerate(group_index):
            for did, ls in zip(item["dev"], group_losses[i].tolist()):
                baseline_collect[did].append(ls)
        self.baseline = {
            k: (float(np.mean(v)), float(np.std(v)) if np.std(v) != 0 else 1.0)
            for k, v in baseline_collect.items()
        }

        self.dev_list = list(self.batched.keys())

    def __len__(self):
        return len(self.dev_list)

    def _encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_len
        )["input_ids"][0]
        return ids

    def __getitem__(self, index: int):
        dev_id = self.dev_list[index]
        batch_data = self.batched[dev_id]

        # All pairs in this dev_id share the same dev split — take first.
        dev_indices: List[int] = batch_data[0]["dev"]
        dev_local = list(range(len(dev_indices)))
        np.random.shuffle(dev_local)
        dev_local = dev_local[: self.reference_size]
        dev_ids_used = [dev_indices[j] for j in dev_local]

        dev_inputs = [self._encode(self.dev_texts_by_id[dev_id][j]) for j in dev_local]

        # Sample topk subsets from this dev_id group.
        set_idx = list(range(len(batch_data)))
        np.random.shuffle(set_idx)
        set_idx = set_idx[: self.topk]

        set_inputs: List[List[torch.Tensor]] = []
        targets: List[List[float]] = []
        for i in set_idx:
            item = batch_data[i]
            pair_idx = item["_pair_idx"]
            subset_texts = self.subset_texts[pair_idx]
            set_inputs.append([self._encode(t) for t in subset_texts])

            normalized = [
                (self.baseline[did][0] - ls) / self.baseline[did][1]
                for ls, did in zip(item["loss"], item["dev"])
            ]
            targets.append([normalized[j] for j in dev_local])

        # targets shape: (topk, reference_size) -> transpose to (reference_size, topk)
        targets_t = torch.tensor(targets, dtype=torch.float32).t().tolist()
        return set_inputs, dev_inputs, targets_t


def _collate(batch):
    set_input, dev_input, targets = zip(*batch)
    set_input = sum(set_input, [])  # flatten across batch
    dev_input = sum(dev_input, [])
    targets = sum(targets, [])

    features = {
        "dev_input": pad_sequence(dev_input, batch_first=True, padding_value=0),
        "targets": torch.clip(torch.tensor(targets, dtype=torch.float32), min=-10.0, max=10.0),
    }
    for i, x in enumerate(set_input):
        features[f"set_input_{i}"] = pad_sequence(x, batch_first=True, padding_value=0)
    return features


class AirRepTrainer:
    def __init__(
        self,
        base_model: str = "thenlper/gte-small",
        batch_size: int = 1,
        epochs: int = 10,
        lr: float = 1e-4,
        topk: int = 32,
        reference_size: int = 256,
        max_len: int = 512,
        save_path: str = "./checkpoints/airrep",
        device: Optional[str] = None,
    ):
        self.base_model = base_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.topk = topk
        self.reference_size = reference_size
        self.max_len = max_len
        self.save_path = save_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self, tokenizer):
        base_encoder = AutoModel.from_pretrained(
            self.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        config = AirRepConfig(**base_encoder.config.to_dict())
        config.model_type = "airrep"
        model = AirRepModel(config)
        model.bert = base_encoder
        # Resize for tokenizer if needed (gte-small uses bert tokenizer; usually matches).
        return model.to(self.device)

    def train(
        self,
        subset_texts: List[List[str]],
        dev_texts_by_id: Dict[int, List[str]],
        group_losses: torch.Tensor,
        group_index: List[Dict],
    ) -> AirRepModel:
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = self._build_model(tokenizer)

        dataset = AirRepDataset(
            subset_texts=subset_texts,
            dev_texts_by_id=dev_texts_by_id,
            tokenizer=tokenizer,
            group_losses=group_losses,
            group_index=group_index,
            topk=self.topk,
            reference_size=self.reference_size,
            max_len=self.max_len,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate, num_workers=0
        )

        optimizer = AdamW(model.parameters(), lr=self.lr)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)

        loss_window: List[float] = []
        for epoch in range(self.epochs):
            print(f"AirRep epoch {epoch + 1}/{self.epochs}")
            model.train()
            pbar = tqdm(loader)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                dev_embed = model(input_ids=batch["dev_input"])

                targets = batch["targets"]
                scores_cols = []
                # First pass: encode subsets w/o grads to build the full score matrix.
                for i in range(self.topk + 10):
                    key = f"set_input_{i}"
                    if key not in batch:
                        break
                    with torch.no_grad():
                        set_embed = model(input_ids=batch[key])
                    s = dev_embed @ set_embed.t()
                    attn = F.softmax(s.abs(), dim=-1)
                    s = (attn * s).sum(dim=-1, keepdim=True).float()
                    scores_cols.append(s)
                if not scores_cols:
                    continue
                scores = torch.cat(scores_cols, dim=1)

                loss = ranknet_loss(scores, targets)
                loss.backward()

                # Second pass: re-encode each subset with grads, refresh that column, accumulate grads.
                dev_embed = dev_embed.detach()
                for i in range(self.topk + 10):
                    key = f"set_input_{i}"
                    if key not in batch:
                        break
                    scores = scores.detach()
                    set_embed = model(input_ids=batch[key])
                    s = dev_embed @ set_embed.t()
                    attn = F.softmax(s.abs(), dim=-1)
                    s = (attn * s).sum(dim=-1, keepdim=True).float()
                    scores[:, i] = s.flatten()
                    pass_loss = ranknet_loss(scores, targets)
                    pass_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                loss_window.append(loss.item())
                pbar.set_postfix(loss=float(np.mean(loss_window[-100:])))

        # Save final model.
        model.save_pretrained(self.save_path)
        tokenizer.save_pretrained(self.save_path)
        return model
