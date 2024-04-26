from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from wenet.LLM.pattern import Pattern
from wenet.text.hugging_face_tokenizer import HuggingFaceTokenizer
from wenet.utils.common import IGNORE_ID


def parse_sft(
    sample,
    tokenizer: HuggingFaceTokenizer,
    template: Pattern,
    add_bos: bool = True,
    add_eos: bool = True,
):
    """Paser sft json line to tensor

        Args:
            sample:
            {
                'system': 'you are a helpful ...',
                "conversation": [{
                    'human': '...',
                    'assistant': '...'
                }]
            }

        Returns:
            {input_ids, output_ids}
    """
    chat_pattern = template
    input_ids = []
    output_ids = []
    system_text = sample.get('system', '')
    if chat_pattern.system is not None:
        system_text = chat_pattern.system.format(content=system_text)
        if add_bos:
            system_text = template.bos + system_text
        _, system_text_ids = tokenizer.tokenize(system_text)
        input_ids += system_text_ids
        output_ids += [IGNORE_ID] * len(system_text_ids)
    conversations = sample['conversation']
    assert isinstance(conversations, List)
    for conversation in conversations:
        human = conversation['human']
        human = chat_pattern.user.format(content=human)
        _, human_ids = tokenizer.tokenize(human)
        input_ids += human_ids
        output_ids += [IGNORE_ID] * len(human_ids)
        if 'assistant' in conversation:
            assistant = conversation['assistant']
            assistant = chat_pattern.assistant.format(content=assistant)
            _, assistant_ids = tokenizer.tokenize(assistant)
            input_ids += assistant_ids
            output_ids += assistant_ids

    if add_eos:
        eos_id = tokenizer.tokens2ids([template.eos])
        input_ids += eos_id
        output_ids += eos_id

    assert len(input_ids) == len(output_ids)
    return {
        'input_ids': torch.tensor(input_ids),
        'output_ids': torch.tensor(output_ids),
    }


def parse_pretrain(sample,
                   tokenizer: HuggingFaceTokenizer,
                   template: Pattern,
                   add_bos: bool = True,
                   add_eos: bool = False):
    """ Parse text from json line

        Args:
            sample: str, str is a json line has txt

        Returns:
            {input_ids, output_ids}
    """
    assert 'text' in sample
    text = sample['text']
    _, input_ids = tokenizer.tokenize(text)
    if add_bos:
        input_ids = [template.bos] + input_ids
    if add_eos:
        input_ids = input_ids + [template.eos]

    return {
        'input_ids': torch.tensor(input_ids),
        'output_ids': torch.tensor(input_ids),
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
    sorted_labels = [sample[i]['output_ids'] for i in order]
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
        "target_lengths": feats_lengths,
    }
    return batch
