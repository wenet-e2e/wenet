from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from wenet.AudioLLM.template import Template
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer
from wenet.utils.common import IGNORE_ID

def parse_audiosft(sample,
                   tokenizer: HuggingFaceTokenizer,
                   template: Template,
                   add_bos: bool = True,
                   add_eos: bool = True):
    """Paser sft json line to tensor

        Args:
            sample:
            {
                'system': 'you are a helpful ...',
                'text': '...',
                'wav': '...',
            }

        Returns:
            {input_ids, output_ids}
    """
    chat_pattern = template
    prefix_input_ids = []
    prefix_output_ids = []
    suffix_input_ids = []
    suffix_output_ids = []
    system_text = sample.get('system', 'you are a helpful ASR agent')
    if chat_pattern.system is not None:
        system_text = chat_pattern.system.format(content=system_text)
        if add_bos:
            system_text = template.bos + system_text
        _, system_text_ids = tokenizer.tokenize(system_text)
        prefix_input_ids += system_text_ids
        prefix_output_ids += [IGNORE_ID] * len(system_text_ids)

    human_text = sample.get('human_text', 'please help me transcribe this audio in english')
    human_text = chat_pattern.prefix_user.format(content=human_text)
    _, human_ids = tokenizer.tokenize(human_text)
    prefix_input_ids += human_ids
    prefix_output_ids += [IGNORE_ID] * len(human_ids)

    _, suffix_ids = tokenizer.tokenize(chat_pattern.suffix_user)
    suffix_input_ids += suffix_ids
    suffix_output_ids += [IGNORE_ID] * len(suffix_ids)

    assistant = sample['txt']
    assistant = chat_pattern.assistant.format(content=assistant)
    _, assistant_ids = tokenizer.tokenize(assistant)
    suffix_input_ids += assistant_ids
    suffix_output_ids += assistant_ids

    if add_eos:
        eos_id = tokenizer.tokens2ids([template.eos])
        suffix_input_ids += eos_id
        suffix_output_ids += eos_id

    assert len(prefix_input_ids) == len(prefix_output_ids)
    assert len(suffix_input_ids) == len(suffix_output_ids)

    sample['prefix_input_ids'] = torch.tensor(prefix_input_ids)
    sample['prefix_output_ids'] = torch.tensor(prefix_output_ids)
    sample['suffix_input_ids'] = torch.tensor(suffix_input_ids)
    sample['suffix_output_ids'] = torch.tensor(suffix_output_ids)
    return sample


def shift(sample):
    prefix_output_ids = sample['prefix_output_ids']
    suffix_input_ids = sample['suffix_input_ids']

    sample['prefix_output_ids'] = prefix_output_ids[1:]
    sample['suffix_input_ids'] = suffix_input_ids[:-1]
    return sample


def filter(sample, token_max_length: int = 8190, token_min_length=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {input_ids, output_ids}
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length

        Returns:
            bool: True to keep, False to filter
    """
    total_lens = sample['prefix_input_ids'].size(0) + \
                 sample['feat'].size(0) + \
                 sample['suffix_input_ids'].size(0)
    if total_lens < token_min_length:
        return False
    if total_lens > token_max_length:
        return False
    return True


def sort_by_input(sample):
    total_lens = sample['prefix_input_ids'].size(0) + \
                 sample['feat'].size(0) + \
                 sample['suffix_input_ids'].size(0)
    return total_lens

def input_length_fn(sample) -> int:
    total_lens = sample['prefix_input_ids'].size(0) + \
                 sample['feat'].size(0) + \
                 sample['suffix_input_ids'].size(0)
    return total_lens

def padding(data: List[Dict]):
    """ Padding the data into training data

        Args:
            data: List[{input_ids, output_ids}

        Returns:
            Tuple(feats, labels)
    """
    sample = data
    
    total_lens = torch.tensor([x['prefix_input_ids'].size(0) + 
                               x['feat'].size(0) +
                               x['suffix_input_ids'].size(0)
                               for x in sample],dtype=torch.int32)

    order = torch.argsort(total_lens, descending=True)
    sorted_keys = [sample[i]['key'] for i in order]
    prefix_tokens_lengths = torch.tensor(
        [sample[i]['prefix_input_ids'].size(0) for i in order], dtype=torch.int32)
    sorted_prefix_tokens = [sample[i]['prefix_input_ids'] for i in order]
    sorted_prefix_labels = [sample[i]['prefix_output_ids'] for i in order]
    padded_prefix_tokens = pad_sequence(sorted_prefix_tokens,
                                batch_first=True,
                                padding_value=0)
    padding_prefix_labels = pad_sequence(sorted_prefix_labels,
                                  batch_first=True,
                                  padding_value=IGNORE_ID)
    audio_feats_lengths = torch.tensor(
        [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
    sorted_audio_feats = [sample[i]['feat'] for i in order]
    padded_audio_feats = pad_sequence(sorted_audio_feats,
                                batch_first=True,
                                padding_value=0)
    suffix_tokens_lengths = torch.tensor(
        [sample[i]['suffix_input_ids'].size(0) for i in order], dtype=torch.int32)
    sorted_suffix_tokens = [sample[i]['suffix_input_ids'] for i in order]
    sorted_suffix_labels = [sample[i]['suffix_output_ids'] for i in order]
    padded_suffix_tokens = pad_sequence(sorted_suffix_tokens,
                                batch_first=True,
                                padding_value=0)
    padding_suffix_labels = pad_sequence(sorted_suffix_labels,
                                  batch_first=True,
                                  padding_value=IGNORE_ID)
    batch = {
        'keys': sorted_keys,
        'prefix_tokens': padded_prefix_tokens,
        'audio_feats': padded_audio_feats,
        'suffix_tokens': padded_suffix_tokens,
        "prefix_target": padding_prefix_labels,
        "suffix_target": padding_suffix_labels,
        "prefix_tokens_lengths": prefix_tokens_lengths,
        "audio_feats_lengths": audio_feats_lengths,
        "suffix_tokens_lengths": suffix_tokens_lengths,
        "target_lengths": prefix_tokens_lengths + audio_feats_lengths + suffix_tokens_lengths
    }
    return batch
