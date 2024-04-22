import dataclasses
from typing import Dict, Union


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
    # rope style: google or llama
    rope_style: str = 'google'
    # rms_norm offset
    rms_norm_offset: bool = True
    # activation type
    activation_type: str = 'gelu'
    # gelu approximate
    gelu_approximate: Union[str, None] = None
    # The dtype of the weights.
    dtype: str = 'bfloat16'

    def to_wenet_config(self) -> Dict:
        configs = {}
        configs['max_position_embeding'] = self.max_position_embeddings
        configs['num_blocks'] = self.num_hidden_layers
        configs['attention_heads'] = self.num_attention_heads
        configs['n_kv_head'] = self.num_key_value_heads
        configs['head_dim'] = self.head_dim
        configs['hidden_size'] = self.hidden_size
        configs['linear_units'] = self.intermediate_size
        configs['norm_eps'] = self.rms_norm_eps
        configs['rope_theta'] = self.rope_theta
        configs['activation_type'] = self.activation_type
        configs['gelu_approximate'] = self.gelu_approximate
        configs['rope_style'] = self.rope_style
        configs['rms_norm_offset'] = self.rms_norm_offset
        return configs


def gemma_config_for_7b() -> Config:
    return Config(rope_theta=10000.0, gelu_approximate='tanh')


def gemma_config_for_2b() -> Config:
    return Config(num_hidden_layers=18,
                  num_attention_heads=8,
                  num_key_value_heads=1,
                  hidden_size=2048,
                  intermediate_size=16384,
                  rope_theta=10000.0,
                  gelu_approximate='tanh')


def llama3_config_for_8b() -> Config:
    return Config(vocab_size=128256,
                  num_hidden_layers=32,
                  hidden_size=4096,
                  num_attention_heads=32,
                  num_key_value_heads=8,
                  head_dim=128,
                  intermediate_size=14336,
                  rms_norm_eps=1e-5,
                  rope_theta=500000.0,
                  activation_type='swish',
                  rms_norm_offset=False,
                  rope_style='llama')


def llama3_config_for_70b() -> Config:
    return Config(vocab_size=128256,
                  num_hidden_layers=80,
                  hidden_size=8192,
                  head_dim=128,
                  num_attention_heads=64,
                  num_key_value_heads=8,
                  intermediate_size=28672,
                  rms_norm_eps=1e-5,
                  rope_theta=500000.0,
                  activation_type='swish',
                  rms_norm_offset=False,
                  rope_style='llama')
