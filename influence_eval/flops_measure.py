"""Wrap GPU workloads in torch.utils.flop_counter.FlopCounterMode to measure
total FLOPs, and persist per-phase counts to JSON for multi-script methods
(e.g. influcoder, which spans gradient_stocking + train + inference).

Convention: each script that does real GPU work runs its main workload under
`flop_counter()` and saves the resulting int into its `{name}_params.pt` as
`measured_flops`. Multi-phase methods accumulate via `_flops.json` files
read/written with `load_phase_flops`/`save_phase_flops`.
"""

import json
import os
from contextlib import contextmanager
from typing import Iterator

from torch.utils.flop_counter import FlopCounterMode


@contextmanager
def flop_counter() -> Iterator[FlopCounterMode]:
    """Yields a FlopCounterMode. Use `counter.get_total_flops()` after exit."""
    mode = FlopCounterMode(display=False)
    with mode:
        yield mode


def save_phase_flops(path: str, flops: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"flops": int(flops)}, f)


def load_phase_flops(path: str) -> int:
    """Return the saved flop count, or 0 if the file does not exist."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return int(json.load(f)["flops"])


def add_phase_flops(path: str, flops: int) -> int:
    """Accumulate into an existing _flops.json. Returns the new total."""
    total = load_phase_flops(path) + int(flops)
    save_phase_flops(path, total)
    return total
