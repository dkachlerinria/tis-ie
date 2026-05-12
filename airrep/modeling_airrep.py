"""AirRep model: BERT encoder + projection head with influence-scoring API.

Vendored verbatim from AirRep-main/airrep/modeling_airrep.py. The wrapper
class `AirRep` exposes a SentenceTransformer-compatible `.encode(...)` so
tis-ie's shared `compute_train_embeddings` works with it unchanged.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig, BertModel, PreTrainedModel


def mean_pooling(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class AirRepConfig(BertConfig):
    model_type = "airrep"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class AirRepModel(PreTrainedModel):
    config_class = AirRepConfig
    base_model_prefix = "airrep"

    def __init__(self, config: AirRepConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.projector = nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.bfloat16)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(0)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        return self.projector(pooled)


class AirRep:
    """Inference wrapper around AirRepModel with a sentence-transformer-style API."""

    def __init__(
        self,
        model: AirRepModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        config = AirRepConfig.from_pretrained(model_name_or_path)
        model = AirRepModel.from_pretrained(
            model_name_or_path, config=config, torch_dtype=torch.bfloat16
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        return cls(model, tokenizer, device)

    @staticmethod
    def _format_item(item: Dict[str, Any]) -> str:
        prompt = "Question: " + " ".join(item["input"].split()[:256]) + "\nAnswer:"
        return prompt + " " + item["output"]

    def encode(
        self,
        texts: Union[str, List[str], List[Dict[str, Any]]],
        batch_size: int = 128,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        if texts and isinstance(texts[0], dict):
            texts = [self._format_item(item) for item in texts]

        all_embeds = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeds = self.model(**inputs)
                if normalize_embeddings:
                    embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())

        result = torch.cat(all_embeds, dim=0)
        if convert_to_numpy:
            result = result.float().numpy()
        return result

    def similarity(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        corpus_embeddings: Union[np.ndarray, torch.Tensor],
        softmax: bool = True,
        chunk_size: int = 1000,
    ) -> Union[np.ndarray, torch.Tensor]:
        is_numpy_input = isinstance(query_embeddings, np.ndarray)
        if is_numpy_input:
            query_embeddings = torch.from_numpy(query_embeddings).float()
        if isinstance(corpus_embeddings, np.ndarray):
            corpus_embeddings = torch.from_numpy(corpus_embeddings).float()

        num_corpus = corpus_embeddings.shape[0]
        all_scores = []
        for i in range(0, num_corpus, chunk_size):
            end_i = min(i + chunk_size, num_corpus)
            all_scores.append(query_embeddings @ corpus_embeddings[i:end_i].T)
        scores = torch.cat(all_scores, dim=-1)

        if softmax:
            attn = torch.softmax(scores.abs(), dim=-1)
            weighted_scores = (attn * scores).sum(dim=-1)
        else:
            weighted_scores = scores.sum(dim=-1)

        return weighted_scores.numpy() if is_numpy_input else weighted_scores

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
