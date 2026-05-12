"""AirRep selection method vendored into tis-ie.

Public API mirrors AirRep-main: AirRepModel/AirRepConfig/AirRep (modeling),
SubsetDevSampler (pair sampling), SFTTrainer (stage-2), AirRepTrainer (stage-3).
"""

from .modeling_airrep import AirRep, AirRepConfig, AirRepModel
from .data_sampler import SubsetDevSampler
from .sft_trainer import SFTTrainer
from .airrep_trainer import AirRepTrainer

__all__ = [
    "AirRep",
    "AirRepConfig",
    "AirRepModel",
    "SubsetDevSampler",
    "SFTTrainer",
    "AirRepTrainer",
]
