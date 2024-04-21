import json
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from wenet.LLM.pattern import WENET_LLM_PATTERN
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.utils.common import IGNORE_ID


def parse_sft(sample, tokenizer: BaseTokenizer, style='gemma'):
    """paser sft json line to tensor
       sample:
        {
            'system': 'you are a helpful ...',
            "conversation": [{
                'human': '...',
                'assistant': '...'
            }]
        }
    """
    sample = json.loads(sample)
    chat_pattern = WENET_LLM_PATTERN[style]
    input_ids = []
    output_mask = []
    system_text = sample.get('system', '')
    if chat_pattern.system_format is not None:
        system_text = chat_pattern.system_format.format(content=system_text)
        _, system_text_ids = tokenizer.tokenize(system_text)
        input_ids += system_text_ids
        output_mask += [0] * len(system_text_ids)
    conversations = sample['conversation']
    assert isinstance(conversations, List)

    for conversation in enumerate(conversations):
        human = conversation['human']
        assistant = conversation['assistant']

        human = chat_pattern.user_format.format(content=human)
        assistant = chat_pattern.assistant_format.format(content=assistant)

        _, human_ids = tokenizer.tokenize(human)
        _, assistant_ids = tokenizer.tokenize(assistant)

        input_ids += human_ids
        input_ids += assistant_ids
        output_mask += [0] * len(human_ids) + [1] * len(assistant_ids)

    assert len(input_ids) == len(output_mask)
    input_ids_tensor = torch.tensor(input_ids)
    output_mask_tensor = torch.tensor(output_mask)
    output_ids_tensor = torch.where(output_mask_tensor == 0, IGNORE_ID,
                                    input_ids_tensor)
    return {
        'input_ids': input_ids_tensor,
        'output_ids': output_ids_tensor,
    }


def shift(sample):
    input_ids = sample['input_ids']
    output_ids = sample['output_ids']

    sample['input_ids'] = input_ids[:-1]
    sample['output_ids'] = output_ids[1:]
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
    assert 'input_ids' in sample
    assert 'output_ids' in sample
    assert isinstance(sample['input_ids'], torch.Tensor)
    assert isinstance(sample['output_ids'], torch.Tensor)
    if sample['input_ids'].size(0) < token_min_length:
        return False
    if sample['input_ids'].size(0) > token_max_length:
        return False
    return True


def sort_by_input(sample):
    assert 'input_ids' in sample
    assert isinstance(sample['input_ids'], torch.Tensor)
    return sample['input_ids'].size(0)


def input_length_fn(sample) -> int:
    assert 'input_ids' in sample
    return sample['input_ids'].size(0)


def padding(data: List[Dict]):
    """ Padding the data into training data

        Args:
            data: List[{input_ids, output_ids}

        Returns:
            Tuple(feats, labels)
    """
    sample = data
    feats_length = torch.tensor([x['input_ids'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor(
        [sample[i]['input_ids'].size(0) for i in order], dtype=torch.int32)
    sorted_feats = [sample[i]['input_ids'] for i in order]
    sorted_labels = [
        torch.tensor(sample[i]['output_ids'], dtype=torch.int64) for i in order
    ]
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    padding_labels = pad_sequence(sorted_labels,
                                  batch_first=True,
                                  padding_value=-1)

    batch = {
        'feats': padded_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
    }
    return batch


class DynamicBatchWindow:

    def __init__(self, max_tokens_in_batch=22000):
        self.longest_tokens = 0
        self.max_token_in_batch = max_tokens_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'input_ids' in sample
        assert isinstance(sample['input_ids'], torch.Tensor)
        new_tokens = sample['input_ids'].size(0)
        self.longest_tokens = max(self.longest_frames, new_tokens)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_token_in_batch:
            self.longest_frames = new_tokens
            return True
        return False
