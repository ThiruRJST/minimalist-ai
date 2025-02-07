from dataclasses import dataclass


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
    

"""
class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None
    # Special token config
    eos_id: int | None = EOS_ID

"""