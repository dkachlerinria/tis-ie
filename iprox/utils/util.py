import numpy as np
import random
import torch
from transformers import set_seed

# NOTE: Hardcoded chat templates were removed. The pipeline now uses each
# model's native tokenizer.chat_template, with a ChatML fallback applied via
# formatting.ensure_chat_template for base models that ship no template.
# This keeps the format consistent across the pipeline for any given model
# (Qwen2.5/3, Llama-3.x, Gemma, etc.) without locking us to one family.


def setseed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    set_seed(seed)