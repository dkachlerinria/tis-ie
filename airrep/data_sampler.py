"""Sample (subset, dev) pairs for AirRep stage-2 SFT.

Adapted from AirRep-main/airrep/data_sampler.py for tis-ie's setup:
- `select` indices point into the Tulu train pool (`Harvard-DCML/tulu-v2-197K-processed`).
- `dev` indices point into the BBH partition reserved for AirRep dev
  (`bbh_data.load_bbh_samples(start_index=NUM_ANCHORS, n_samples=dev_pool_size)`).
"""

from typing import Dict, List

import numpy as np
from tqdm import tqdm


class SubsetDevSampler:
    def __init__(
        self,
        train_pool_size: int,
        dev_pool_size: int,
        subset_size: int = 1000,
        dev_size: int = 256,
        n_splits: int = 100,
        n_subsets_per_split: int = 100,
        seed: int = 42,
    ):
        if subset_size > train_pool_size:
            raise ValueError(f"subset_size={subset_size} > train_pool_size={train_pool_size}")
        if dev_size > dev_pool_size:
            raise ValueError(f"dev_size={dev_size} > dev_pool_size={dev_pool_size}")
        self.train_pool_size = train_pool_size
        self.dev_pool_size = dev_pool_size
        self.subset_size = subset_size
        self.dev_size = dev_size
        self.n_splits = n_splits
        self.n_subsets_per_split = n_subsets_per_split
        self.seed = seed

    def sample(self) -> List[Dict]:
        rng = np.random.default_rng(self.seed)
        train_idx = np.arange(self.train_pool_size)
        dev_idx = np.arange(self.dev_pool_size)

        pairs: List[Dict] = []
        dev_seen, subset_seen = set(), set()

        for split_id in tqdm(range(self.n_splits), desc="splits"):
            dev_perm = dev_idx.copy()
            rng.shuffle(dev_perm)
            dev_set = sorted(dev_perm[: self.dev_size].tolist())
            dev_key = tuple(dev_set)
            if dev_key in dev_seen:
                continue
            dev_seen.add(dev_key)

            sampled = 0
            attempts = 0
            while sampled < self.n_subsets_per_split and attempts < self.n_subsets_per_split * 5:
                attempts += 1
                train_perm = train_idx.copy()
                rng.shuffle(train_perm)
                select_set = sorted(train_perm[: self.subset_size].tolist())
                key = (split_id, tuple(select_set))
                if key in subset_seen:
                    continue
                subset_seen.add(key)
                pairs.append({
                    "id": f"{split_id}-{sampled}",
                    "select": select_set,
                    "dev": dev_set,
                    "dev_id": split_id,
                })
                sampled += 1

        return pairs
