import dataclasses
from typing import Dict, Optional, Union

import yaml


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

    # scale embed
    scale_embed: bool = True

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
        configs['scale_embed'] = self.scale_embed
        return configs


def wenet_llm_tokenizer_conf(config: Config, tokenizer_path: str,
                             special_tokens: Dict) -> Dict:
    configs = {}
    configs['tokenizer'] = 'huggingface'
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['model'] = tokenizer_path
    configs['tokenizer_conf']['special_tokens'] = special_tokens
    return configs


def wenet_llm_dataset_and_train_conf(config: Config,
                                     template: str = 'gemma') -> Dict:
    configs = {}
    configs['dataset'] = "llm"
    configs['dataset_conf'] = {}
    configs['dataset_conf']['filter_conf'] = {}
    configs['dataset_conf']['filter_conf'][
        'token_max_length'] = config.max_position_embeddings
    configs['dataset_conf']['filter_conf']['token_min_length'] = 1
    configs['dataset_conf']['shuffle'] = True
    configs['dataset_conf']['shuffle_conf'] = {}
    configs['dataset_conf']['shuffle_conf']['shuffle_size'] = 1500
    configs['dataset_conf']['shuffle_list'] = True
    configs['dataset_conf']['shuffle_list_conf'] = {}
    configs['dataset_conf']['shuffle_list_conf']['shuffle_size'] = 15000
    configs['dataset_conf']['sort'] = True
    configs['dataset_conf']['sort_conf'] = {}
    configs['dataset_conf']['sort_conf']['sort_size'] = 500
    configs['dataset_conf']['batch_conf'] = {}
    configs['dataset_conf']['batch_conf']['batch_type'] = 'dynamic'
    configs['dataset_conf']['batch_conf']['max_frames_in_batch'] = 12000

    configs['dataset_conf']['data_style'] = 'sft'
    configs['dataset_conf']['data_style_conf'] = {}
    configs['dataset_conf']['data_style_conf']['add_bos'] = True
    configs['dataset_conf']['data_style_conf']['add_eos'] = True
    configs['dataset_conf']['data_style_conf']['template'] = template
    configs['dataset_conf']['shift'] = True

    configs['grad_clip'] = 5
    configs['accum_grad'] = 4
    configs['max_epoch'] = 100
    configs['log_interval'] = 100
    configs['save_interval'] = 3000

    configs['optim'] = "adam"
    configs['optim_conf'] = {}
    configs['optim_conf']['lr'] = 0.0005
    configs['scheduler'] = "warmuplr"
    configs['scheduler_conf'] = {}
    configs['scheduler_conf']['warmup_steps'] = 12000
    return configs


def wenet_decoderonly_conf(config: Config):
    configs = {}
    configs['decoder'] = 'decoder_only'
    configs['decoder_conf'] = config.to_wenet_config()
    configs['decoder_conf']['dropout_rate'] = 0.0
    configs['decoder_conf']['attention_dropout_rate'] = 0.0
    configs['decoder_conf']['positional_dropout_rate'] = 0.0
    configs['decoder_conf']['gradient_checkpointing'] = True
    configs['decoder_conf']['normalize_before'] = True
    configs['decoder_conf']['use_sdpa'] = True
    return configs


def wenet_model_conf(config: Config, tie_word_embedding: bool = True):
    configs = {}
    configs['output_dim'] = config.vocab_size
    configs['model'] = "causal_lm"
    configs['model_conf'] = {}
    configs['model_conf']['linear_bias'] = False
    configs['model_conf']['tie_word_embedding'] = tie_word_embedding
    configs['model_conf']['lsm_weight'] = 0.1
    configs['model_conf']['length_normalized_loss'] = False
    return configs


def convert_to_wenet_yaml(config: Config,
                          wenet_yaml_path: str,
                          tokenizer_path,
                          template: str = 'gemma',
                          tie_word_embedding: bool = True,
                          special_tokens: Optional[Dict] = None):
    configs = {}
    configs.update(
        wenet_llm_tokenizer_conf(config, tokenizer_path, special_tokens))
    configs.update(wenet_decoderonly_conf(config))
    configs.update(
        wenet_model_conf(config, tie_word_embedding=tie_word_embedding))
    configs.update(wenet_llm_dataset_and_train_conf(config, template=template))

    with open(wenet_yaml_path, '+w') as f:
        f.write(yaml.dump(configs))
        f.flush()

    print(configs)


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
                  rope_style='llama',
                  scale_embed=False)


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
                  rope_style='llama',
                  scale_embed=False)
