from typing import Optional
from wenet.dataset.dataset import Dataset as ASRDatast
from wenet.dataset.llm_dataset import Dataset as LLMDataset
from wenet.text.base_tokenizer import BaseTokenizer


def init_dataset(data_type,
                 conf,
                 data_list_file,
                 tokenizer: Optional[BaseTokenizer] = None,
                 partition=True):
    dataset_type = conf.get('dataset', 'asr')
    if dataset_type == 'asr':
        return ASRDatast(data_type, data_list_file, conf, tokenizer, partition)
    else:
        assert dataset_type == 'llm'
        return LLMDataset(data_type, data_list_file, conf, tokenizer,
                          partition)
