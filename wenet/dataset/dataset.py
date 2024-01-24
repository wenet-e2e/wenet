from functools import partial
from typing import Optional
from wenet.dataset import processor
from wenet.dataset.datapipes import (WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.text.base_tokenizer import BaseTokenizer


def Dataset(data_type,
            data_list_file,
            tokenizer: Optional[BaseTokenizer] = None,
            conf=None,
            partition=True):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
            tokenizer (BaseTokenizer or None): tokenizer to tokenize
    """
    # NOTE(Mddct): partition is always true for datapipe
    _ = partition
    assert conf is not None
    assert data_type in ['raw', 'shard']
    if data_type == 'raw':
        dataset = WenetRawDatasetSource(data_list_file)
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file)
    dataset = dataset.map(processor.decode_wav)

    if tokenizer is not None:
        dataset = dataset.map(partial(processor.tokenize, tokenizer=tokenizer))

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))

    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    speed_perturb = conf.get('speed_perturb', False)
    if speed_perturb:
        dataset = dataset.map(partial(processor.speed_perturb))

    feats_type = conf.get('feats_type', 'fbank')
    assert feats_type in ['fbank', 'mfcc', 'log_mel_spectrogram']
    if feats_type == 'fbank':
        fbank_conf = conf.get('fbank_conf', {})
        dataset = dataset.map(partial(processor.compute_fbank, **fbank_conf))
    elif feats_type == 'mfcc':
        mfcc_conf = conf.get('mfcc_conf', {})
        dataset = dataset.map(partial(processor.compute_mfcc, **mfcc_conf))
    elif feats_type == 'log_mel_spectrogram':
        log_mel_spectrogram_conf = conf.get('log_mel_spectrogram_conf', {})
        dataset = dataset.map(
            partial(processor.compute_log_mel_spectrogram,
                    **log_mel_spectrogram_conf))
    spec_aug = conf.get('spec_aug', True)
    spec_sub = conf.get('spec_sub', False)
    spec_trim = conf.get('spec_trim', False)
    if spec_aug:
        spec_aug_conf = conf.get('spec_aug_conf', {})
        dataset = dataset.map(partial(processor.spec_aug, **spec_aug_conf))
    if spec_sub:
        spec_sub_conf = conf.get('spec_sub_conf', {})
        dataset = dataset.map(partial(processor.spec_sub, **spec_sub_conf))
    if spec_trim:
        spec_trim_conf = conf.get('spec_trim_conf', {})
        dataset = dataset.map(partial(processor.spec_trim, **spec_trim_conf))

    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=processor.padding)
    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.dynamic_batch_window_fn(
                max_frames_in_batch=max_frames_in_batch),
            wrapper_class=processor.padding,
        )

    return dataset
