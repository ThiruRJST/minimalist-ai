from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel

@dataclass
class BLTEntropyModelConfig:
    dim: int = 768
    n_layers: int = 14
    n_heads: int = 12
    max_seqlen: int = 8192
    vocab_size: int = 260
    ffn_dim_multiplier: int = 1.0
    sliding_window: int = 512
    rope_theta: float = 10000.0
    multiple_of: int = 256
    eps: float = 1e-6
    init_std_factor = "disabled"
    init_base_std = None


class PatchingModes(str, Enum):
    entropy = "entropy"
    bpe = "bpe"
    bpe_patcher = "bpe_patcher"
    space = "space"
    static = "static"
    byte = "byte"


class PatcherArgs(BaseModel):
    patching_mode: PatchingModes = PatchingModes.entropy
    patching_device: str = "cuda"
    entropy_model_checkpoint_dir: str | None = None
    realtime_patching: bool = False
    threshold: float = 1.335442066192627
    threshold_add: float | None = None
    max_patch_length: int | None = None
    patch_size: float = 4.5
    patching_batch_size: int = 1
    device: str = "cuda"
    monotonicity: bool = False
    log_time: bool = False
    
    def build(self) -> "Patcher":
        return Patcher(self)