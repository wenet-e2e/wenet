from typing import Optional
from wenet.dataset.llm_dataset import Dataset as LLMDataset
from wenet.text.base_tokenizer import BaseTokenizer


def init_dataset(data_type,
                 conf,
                 data_list_file,
                 tokenizer: Optional[BaseTokenizer] = None,
                 partition=True,
                 dataset_type: str = 'asr'):
    assert dataset_type in ['asr', 'llm']
    conf = conf['dataset_conf']
    if dataset_type == 'asr':
        from wenet.dataset.dataset import Dataset as ASRDatast
        return ASRDatast(data_type, data_list_file, conf, tokenizer, partition)
    else:
        assert tokenizer is not None
        return LLMDataset(data_type, data_list_file, tokenizer, conf,
                          partition)
