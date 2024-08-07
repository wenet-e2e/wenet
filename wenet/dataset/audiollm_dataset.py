from functools import partial
import sys
from wenet.AudioLLM.template import WENET_LLM_Template
from wenet.dataset.datapipes import (WenetRawDatasetSource)
from wenet.dataset import (processor, audiollm_processor)
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.file_utils import read_symbol_table
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer


def Dataset(data_type,
            data_list_file,
            tokenizer: BaseTokenizer,
            conf=None,
            partition=True,
            train=True):
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
        dataset = dataset.map(processor.parse_json)
    else:
        raise NotImplementedError('only support jsonl for now')

    dataset = dataset.map_ignore_error(processor.decode_wav)

    singal_channel_conf = conf.get('singal_channel_conf', {})
    dataset = dataset.map(
        partial(processor.singal_channel, **singal_channel_conf))
    

    speaker_conf = conf.get('speaker_conf', None)
    if speaker_conf is not None:
        assert 'speaker_table_path' in speaker_conf
        speaker_table = read_symbol_table(speaker_conf['speaker_table_path'])
        dataset = dataset.map(
            partial(processor.parse_speaker, speaker_dict=speaker_table))
    if tokenizer is not None:
        dataset = dataset.map(partial(processor.tokenize, tokenizer=tokenizer))
    audio_filter_conf = conf.get('audio_filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **audio_filter_conf))

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

    # TODO: DPO etc
    assert isinstance(tokenizer, HuggingFaceTokenizer)
    style_conf = conf.get('data_style_conf', {})
    template = WENET_LLM_Template[style_conf.get('template', 'audio_llama3')]
    
    dataset = dataset.map(
        partial(
            audiollm_processor.parse_audiosft,
            tokenizer=tokenizer,
            template=template,
            train=train,
            add_bos=style_conf.get('add_bos', True),
            add_eos=style_conf.get('add_eos', True),
        ))
    
    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=audiollm_processor.sort_by_input)
    shift = conf.get('shift', True)
    if shift:
        dataset = dataset.map(audiollm_processor.shift)

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(audiollm_processor.filter, **filter_conf))

    batch_conf = conf.get('batch_conf', {})
    batch_type = batch_conf.get('batch_type', 'static')
    assert batch_type in ['static', 'bucket', 'dynamic']
    if batch_type == 'static':
        assert 'batch_size' in batch_conf
        batch_size = batch_conf.get('batch_size', 16)
        dataset = dataset.batch(
            batch_size,
            wrapper_class=audiollm_processor.padding,
        )
    elif batch_type == 'bucket':
        assert 'bucket_boundaries' in batch_conf
        assert 'bucket_batch_sizes' in batch_conf
        dataset = dataset.bucket_by_sequence_length(
            audiollm_processor.input_length_fn,
            batch_conf['bucket_boundaries'],
            batch_conf['bucket_batch_sizes'],
            wrapper_class=audiollm_processor.padding,
        )
    else:
        max_tokens_in_batch = batch_conf.get('max_tokens_in_batch', 50000)
        dataset = dataset.dynamic_batch(
            processor.DynamicBatchWindow(max_tokens_in_batch),
            wrapper_class=audiollm_processor.padding,
            elem_size_fn=audiollm_processor.input_length_fn,
        )

    return dataset
