import pytest
import torch
from wenet.dataset.dataset import Dataset
from wenet.text.char_tokenizer import CharTokenizer


@pytest.mark.parametrize("params", [
    ("test/resources/dataset/data.list", "test/resources/aishell2.words.txt")
])
def test_dataset(params):
    data_list, unit_table = params[0], params[1]
    data_type = 'raw'
    dataset_conf = {
        'batch_conf': {
            'batch_type': 'dynamic',
            'max_frames_in_batch': 12000
        },
        'fbank_conf': {
            'dither': 0.1,
            'frame_length': 25,
            'frame_shift': 10,
            'num_mel_bins': 80
        },
        'filter_conf': {
            'max_length': 20000,
            'min_length': 0,
            'token_max_length': 200,
            'token_min_length': 1
        },
        'resample_conf': {
            'resample_rate': 16000
        },
        'shuffle': True,
        'shuffle_conf': {
            'shuffle_size': 1500
        },
        'sort': True,
        'sort_conf': {
            'sort_size': 500
        },
        'spec_aug': True,
        'spec_aug_conf': {
            'max_f': 10,
            'max_t': 50,
            'num_f_mask': 2,
            'num_t_mask': 2
        },
        'spec_sub': False,
        'spec_trim': False,
        'speed_perturb': False
    }
    tokenizer = CharTokenizer(unit_table)
    dataset = Dataset(data_type,
                      data_list,
                      tokenizer=tokenizer,
                      conf=dataset_conf)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             num_workers=4,
                                             persistent_workers=True)
    for d in dataloader:
        pass
