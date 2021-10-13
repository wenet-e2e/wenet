from abc import ABC

from dataclasses import dataclass, MISSING, field
from undictify import type_checked_constructor

@dataclass
class HYDRA_BASE(ABC):
    config_name: str = MISSING
    config_path: str = MISSING

    def __init__(self, config_name, config_path):
        self.config_name = config_name
        self.config_path = config_path

@type_checked_constructor()
@dataclass
class DECODING:
    # 'attention', 'ctc_greedy_search',
    # 'ctc_prefix_beam_search', 'attention_rescoring'
    mode: str = 'attention'
    ctc_weight: float = 0.0
    penalty: float = 0.0
    # Specify decoding_chunk_size
    # if it's a unified dynamic chunk trained model,
    # -1 for full chunk
    chunk_size: int = -1
    num_left_chunks: int = -1
    reverse_weight: float = 0.0
    beam_size: int = 10

@type_checked_constructor()
@dataclass
class DECODING_DICT(HYDRA_BASE):
    config: str = MISSING
    test_data: str = MISSING
    checkpoint: str = MISSING
    dict: str = MISSING
    result_file: str = MISSING
    gpu: int = -1  # which gpu goint to be use, -1 means cpu
    data_type: str = 'raw'  # 'raw' or 'shared'
    batch_size: int = 1
    simulate_streaming: bool = False
    decoding : DECODING = field(default_factory=DECODING)
