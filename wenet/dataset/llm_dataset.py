from functools import partial
import sys
from wenet.dataset.datapipes import (WenetRawDatasetSource)
from wenet.dataset import llm_processor
from wenet.text.base_tokenizer import BaseTokenizer


def Dataset(data_type,
            data_list_file,
            tokenizer: BaseTokenizer,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
            partition(bool): whether to do data partition in terms of rank
    """
    assert conf is not None
    assert data_type in ['raw', 'shard']
    # cycle dataset
    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file,
                                        partition=partition,
                                        shuffle=list_shuffle,
                                        shuffle_size=list_shuffle_size,
                                        cycle=cycle)

    else:
        raise NotImplementedError('only support jsonl for now')

    data_style = conf.get('style', 'sft')
    assert data_style in ['pretrain', 'sft']
    data_style_conf = conf.get('style_pattern', 'gemma')
    dataset = dataset.map(
        partial(llm_processor.parse_sft,
                tokenizer=tokenizer,
                style=data_style_conf))
    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=llm_processor.sort_by_input)
    shift = conf.get('shif', True)
    if shift:
        dataset = dataset.map(llm_processor.shift)
    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'bucket', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(
            batch_size,
            wrapper_class=llm_processor.padding,
        )
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            llm_processor.input_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=llm_processor.padding)
    else:
        max_tokens_in_batch = batch_conf.get('max_tokens_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            llm_processor.DynamicBatchWindow(max_tokens_in_batch),
            wrapper_class=llm_processor.padding,
        )

    return dataset
