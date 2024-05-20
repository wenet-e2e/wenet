from typing import Optional
from wenet.dataset.audiollm_dataset import Dataset as AudioLLMDataset
from wenet.dataset.dataset import Dataset as ASRDatast
from wenet.dataset.llm_dataset import Dataset as LLMDataset
from wenet.text.base_tokenizer import BaseTokenizer


def init_dataset(data_type,
                 data_list_file,
                 conf,
                 tokenizer: Optional[BaseTokenizer] = None,
                 partition=True,
                 dataset_type: str = 'asr'):
    assert dataset_type in ['asr', 'llm', 'audio_llm']
    if dataset_type == 'asr':
        return ASRDatast(data_type, data_list_file, tokenizer, conf, partition)
    elif dataset_type == 'audio_llm':
        assert tokenizer is not None
        return AudioLLMDataset(data_type, data_list_file, tokenizer, conf,
                          partition)
    else:
        assert tokenizer is not None
        return LLMDataset(data_type, data_list_file, tokenizer, conf,
                          partition)
