from functools import partial
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from wenet.dataset import processor
from wenet.dataset.datapipes import WenetRawDatasetSource, WenetTarShardDatasetSource


def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "feats_lengths": feats_lengths,
        # NOTE(Mddct): cv need targets , refine later
        "target": padded_feats,
        "target_lengths": feats_lengths,
    }
    return batch


def Dataset(data_type, data_list_file, conf=None, partition=True):
    """ Construct dataset from arguments for ssl model

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw file level. The second is global shuffle
        at training samples level.

        Args:
            data_type(str): raw/shard
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
        dataset = dataset.map(processor.parse_json)
    else:
        dataset = WenetTarShardDatasetSource(data_list_file,
                                             partition=partition,
                                             shuffle=list_shuffle,
                                             shuffle_size=list_shuffle_size,
                                             cycle=cycle)
    dataset = dataset.map_ignore_error(processor.decode_wav)

    singal_channel_conf = conf.get('singal_channel_conf', {})
    dataset = dataset.map(
        partial(processor.singal_channel, **singal_channel_conf))

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
    assert batch_type in ['static', 'bucket', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(batch_size, wrapper_class=padding)
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            processor.feats_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=padding)
    else:
        max_frames_in_batch = batch_conf.get('max_frames_in_batch', 12000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_frames_in_batch),
            wrapper_class=padding,
        )

    return dataset


def init_dataset(data_type, data_list_file, conf=None, partition=True):
    return Dataset(data_type, data_list_file, conf, partition)
