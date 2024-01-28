from functools import partial
import pytest
import torch
from torch.utils.data import datapipes
import torchaudio

from wenet.dataset import processor
from wenet.dataset.datapipes import SortDataPipe  # noqa
from wenet.utils.init_tokenizer import init_tokenizer


@pytest.mark.parametrize("symbol_table_path", [
    "test/resources/librispeech.words.txt", "test/resources/aishell2.words.txt"
])
def test_tokenize(symbol_table_path):
    txts = [{
        "txt": "震东好帅"
    }, {
        "txt": " 吴迪也好帅 "
    }, {
        "txt": "binbin is also handsome"
    }, {
        "txt": " life is short i use wenet "
    }, {
        "txt": "超哥 is the most handsome 吧"
    }, {
        "txt": " 人生苦短i use wenet "
    }, {
        "txt": "人生苦短I USE WENET"
    }, {
        "txt": "zhendong ist so schön"
    }, {
        "txt": " zhendong ist so schön "
    }, {
        "txt": "It's okay"
    }]
    configs = {}
    configs['tokenizer_conf'] = {}
    configs['tokenizer_conf']['symbol_table_path'] = symbol_table_path
    configs['tokenizer_conf']['non_lang_syms_path'] = None
    configs['tokenizer_conf']['split_with_space'] = False
    if symbol_table_path == "test/resources/librispeech.words.txt":
        bpe_model = "test/resources/librispeech.train_960_unigram5000.bpemodel"
        refs = [{
            "tokens": ['震', '东', '好', '帅'],
            "label": [1, 1, 1, 1]
        }, {
            "tokens": ['吴', '迪', '也', '好', '帅'],
            "label": [1, 1, 1, 1, 1]
        }, {
            "tokens": ['▁B', 'IN', 'B', 'IN', '▁IS', '▁ALSO', "▁HANDSOME"],
            "label": [347, 2216, 346, 2216, 2332, 143, 1990]
        }, {
            "tokens":
            ['▁LIFE', '▁IS', '▁SHORT', '▁I', '▁USE', '▁WE', 'NE', 'T'],
            "label": [2568, 2332, 3968, 2152, 4699, 4833, 2926, 4366]
        }, {
            "tokens": ['超', '哥', '▁IS', '▁THE', '▁MOST', '▁HANDSOME', '吧'],
            "label": [1, 1, 2332, 4435, 2860, 1990, 1]
        }, {
            "tokens": ['人', '生', '苦', '短', '▁I', '▁USE', '▁WE', 'NE', 'T'],
            "label": [1, 1, 1, 1, 2152, 4699, 4833, 2926, 4366]
        }, {
            "tokens": ['人', '生', '苦', '短', '▁I', '▁USE', '▁WE', 'NE', 'T'],
            "label": [1, 1, 1, 1, 2152, 4699, 4833, 2926, 4366]
        }, {
            "tokens":
            ['▁', 'Z', 'HEN', 'DO', 'NG', '▁IS', 'T', '▁SO', '▁SCH', 'Ö', 'N'],
            "label":
            [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
        }, {
            "tokens":
            ['▁', 'Z', 'HEN', 'DO', 'NG', '▁IS', 'T', '▁SO', '▁SCH', 'Ö', 'N'],
            "label":
            [3, 4999, 2048, 1248, 2960, 2332, 4366, 4072, 3844, 1, 2901]
        }, {
            "tokens": ['▁IT', "'", 'S', '▁O', 'KA', 'Y'],
            "label": [2344, 2, 3790, 3010, 2418, 4979]
        }]
        configs['tokenizer'] = 'bpe'
        configs['tokenizer_conf']['bpe_path'] = bpe_model
    else:
        bpe_model = None
        refs = [{
            "tokens": ['震', '东', '好', '帅'],
            "label": [4932, 80, 1059, 1375]
        }, {
            "tokens": ['吴', '迪', '也', '好', '帅'],
            "label": [656, 4540, 117, 1059, 1375]
        }, {
            "tokens": [
                'b', 'i', 'n', 'b', 'i', 'n', '▁', 'i', 's', '▁', 'a', 'l',
                's', 'o', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e'
            ],
            "label": [
                9, 23, 33, 9, 23, 33, 1, 23, 43, 1, 7, 29, 43, 35, 1, 21, 7,
                33, 13, 43, 35, 31, 15
            ]
        }, {
            "tokens": [
                'l', 'i', 'f', 'e', '▁', 'i', 's', '▁', 's', 'h', 'o', 'r',
                't', '▁', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e', 'n', 'e', 't'
            ],
            "label": [
                29, 23, 17, 15, 1, 23, 43, 1, 43, 21, 35, 41, 46, 1, 23, 1, 48,
                43, 15, 1, 52, 15, 33, 15, 46
            ]
        }, {
            "tokens": [
                '超', '哥', '▁', 'i', 's', '▁', 't', 'h', 'e', '▁', 'm', 'o',
                's', 't', '▁', 'h', 'a', 'n', 'd', 's', 'o', 'm', 'e', '▁', '吧'
            ],
            "label": [
                4395, 736, 1, 23, 43, 1, 46, 21, 15, 1, 31, 35, 43, 46, 1, 21,
                7, 33, 13, 43, 35, 31, 15, 1, 647
            ]
        }, {
            "tokens": [
                '人', '生', '苦', '短', 'i', '▁', 'u', 's', 'e', '▁', 'w', 'e',
                'n', 'e', 't'
            ],
            "label":
            [155, 2980, 3833, 3178, 23, 1, 48, 43, 15, 1, 52, 15, 33, 15, 46]
        }, {
            "tokens": [
                '人', '生', '苦', '短', 'I', '▁', 'U', 'S', 'E', '▁', 'W', 'E',
                'N', 'E', 'T'
            ],
            "label":
            [155, 2980, 3833, 3178, 24, 1, 49, 44, 16, 1, 53, 16, 34, 16, 47]
        }, {
            "tokens": [
                'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't',
                '▁', 's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
            ],
            "label": [
                58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1,
                43, 11, 21, 1, 33
            ]
        }, {
            "tokens": [
                'z', 'h', 'e', 'n', 'd', 'o', 'n', 'g', '▁', 'i', 's', 't',
                '▁', 's', 'o', '▁', 's', 'c', 'h', 'ö', 'n'
            ],
            "label": [
                58, 21, 15, 33, 13, 35, 33, 19, 1, 23, 43, 46, 1, 43, 35, 1,
                43, 11, 21, 1, 33
            ]
        }, {
            "tokens": ['I', 't', "'", 's', '▁', 'o', 'k', 'a', 'y'],
            "label": [24, 46, 2, 43, 1, 35, 27, 7, 56]
        }]
        configs['tokenizer'] = 'char'

    tokenizer = init_tokenizer(configs)
    outs = [processor.tokenize(txt, tokenizer) for txt in txts]
    for (hyp, ref) in zip(outs, refs):
        assert (len(hyp["tokens"]) == len(ref["tokens"]))
        assert (all(h == r for h, r in zip(hyp["tokens"], ref["tokens"])))
        assert (len(hyp["label"]) == len(ref["label"]))
        assert (all(h == r for h, r in zip(hyp["label"], ref["label"])))


def test_filter():
    input = [
        {
            'wav': torch.rand(1, 10 * 16000),
            'sample_rate': 16000
        },
        {
            'wav': torch.rand(1, 10000 * 16000),
            'sample_rate': 16000
        },
    ]

    dataset = datapipes.iter.IterableWrapper(input)
    dataset = dataset.filter(partial(processor.filter, max_length=1000))
    expected = [input[0]]
    result = []
    for d in dataset:
        result.append(d)

    assert len(expected) == len(result)
    for r, e in zip(result, expected):
        assert r.keys() == e.keys()
        assert torch.allclose(r['wav'], e['wav'])
        assert r['sample_rate'] == e['sample_rate']


@pytest.mark.parametrize("wav_file", [
    "test/resources/aishell-BAC009S0724W0121.wav",
    "test/resources/librispeech-1995-1837-0001.wav",
])
def test_compute_fbank(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file, normalize=False)
    waveform = waveform.to(torch.float)
    assert sample_rate == 16000
    fbank_args = {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
        "energy_floor": 0.0,
        "sample_frequency": 16000
    }
    mat = torchaudio.compliance.kaldi.fbank(waveform=waveform, **fbank_args)

    fbank_args.pop("energy_floor")
    fbank_args.pop("sample_frequency")
    input = {
        'wav': torchaudio.load(wav_file)[0],
        'sample_rate': 16000,
        'key': wav_file,
    }
    assert torch.allclose(
        processor.compute_fbank(input, **fbank_args)['feat'], mat)


def test_sort_by_feats():
    samples = [
        {
            "feat": torch.ones(1000, 80)
        },
        {
            "feat": torch.ones(100, 80)
        },
        {
            "feat": torch.ones(10, 80)
        },
        {
            "feat": torch.ones(1, 80)
        },
    ]
    expected = [
        {
            "feat": torch.ones(1, 80)
        },
        {
            "feat": torch.ones(10, 80)
        },
        {
            "feat": torch.ones(100, 80)
        },
        {
            "feat": torch.ones(1000, 80)
        },
    ]

    dataset = datapipes.iter.IterableWrapper(samples)
    dataset = dataset.sort(key_func=processor.sort_by_feats)

    results = []
    for d in dataset:
        results.append(d)
    assert len(results) == len(samples)
    assert all(
        torch.allclose(r['feat'], h['feat'])
        for (r, h) in zip(expected, results))
