import dataclasses


# https://github.com/google/gemma_pytorch/blob/main/gemma/config.py#L32
@dataclasses.dataclass
class Config:
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # tope theta
    rope_theta: float = 500000.0
    # The dtype of the weights.
    dtype: str = 'bfloat16'


def gemma_config_for_7b() -> Config:
    return Config(rope_theta=10000.0)


def gemma_config_for_2b() -> Config:
    return Config(num_hidden_layers=18,
                  num_attention_heads=8,
                  num_key_value_heads=1,
                  hidden_size=2048,
                  intermediate_size=16384,
                  rope_theta=10000.0)


def llama3_config_for_8b() -> Config:
    return Config(vocab_size=128256,
                  num_hidden_layers=32,
                  hidden_size=4096,
                  num_attention_heads=32,
                  num_key_value_heads=8,
                  head_dim=128,
                  intermediate_size=14336,
                  rms_norm_eps=1e-5,
                  rope_theta=500000.0)


def llama3_config_for_70b() -> Config:
    return Config(vocab_size=128256,
                  num_hidden_layers=80,
                  hidden_size=8192,
                  head_dim=128,
                  num_attention_heads=64,
                  num_key_value_heads=8,
                  intermediate_size=28672,
                  rms_norm_eps=1e-5,
                  rope_theta=500000.0)
