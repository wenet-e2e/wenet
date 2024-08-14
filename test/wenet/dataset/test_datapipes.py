import pytest
import torch
from torch.utils.data import datapipes
from torch.utils.data.datapipes.iter import IterableWrapper
from functools import partial

from wenet.dataset.datapipes import (InterlaveDataPipe, RepeatDatapipe,
                                     SortDataPipe, WenetRawDatasetSource,
                                     WenetTarShardDatasetSource)
from wenet.dataset.processor import (DynamicBatchWindow, decode_wav, padding,
                                     parse_json, compute_fbank,
                                     detect_language, detect_task)


@pytest.mark.parametrize("data_list", [
    "test/resources/dataset/data.list",
])
def test_WenetRawDatasetSource(data_list):

    dataset = WenetRawDatasetSource(data_list)
    expected = []
    with open(data_list, 'r') as fin:
        for line in fin:
            line = line.strip('\n')
            expected.append({"file_name": data_list, "line": line})
    result = []
    for elem in dataset:
        result.append(elem)

    assert len(result) == len(expected)
    for (i, elem) in enumerate(result):
        for key, value in elem.items():
            assert key in expected[i].keys()
            assert value == expected[i][key]


@pytest.mark.parametrize("data_list", [(
    "test/resources/dataset/data.list",
    "test/resources/dataset/data.shards.list",
)])
def test_dataset_consistently(data_list):
    raw_list, tar_list = data_list
    raw_dataset = WenetRawDatasetSource(raw_list)
    raw_dataset = raw_dataset.map(parse_json)
    raw_dataset = raw_dataset.map(decode_wav)
    raw_dataset = raw_dataset.map(compute_fbank)
    raw_results = []
    for d in raw_dataset:
        raw_results.append(d)

    keys = ["key", "txt", "file_name", "wav", "sample_rate", "feat"]
    for r in raw_results:
        assert set(r.keys()) == set(keys)
    tar_dataset = WenetTarShardDatasetSource(tar_list)
    tar_dataset = tar_dataset.map(decode_wav)
    tar_dataset = tar_dataset.map(compute_fbank)
    tar_results = []
    for d in tar_dataset:
        tar_results.append(d)
    keys.append('tar_file_name')
    for r in tar_results:
        assert set(r.keys()) == set(keys)

    assert len(tar_results) == len(raw_results)
    sorted(tar_results, key=lambda elem: elem['key'])
    sorted(raw_results, key=lambda elem: elem['key'])
    same_keys = ["txt", "wav", "sample_rate", "feat"]
    for (i, tar_result) in enumerate(tar_results):
        for k in same_keys:
            if isinstance(tar_result[k], torch.Tensor):
                assert isinstance(raw_results[i][k], torch.Tensor)
                assert torch.allclose(tar_result[k], raw_results[i][k])
            else:
                assert tar_result[k] == raw_results[i][k]


def key_func(elem):
    return elem


def test_sort_datapipe():
    N = 10
    dataset = datapipes.iter.IterableWrapper(range(N))
    dataset = SortDataPipe(dataset, key_func=key_func, reverse=True)
    for (i, d) in enumerate(dataset):
        assert d == N - 1 - i


def fake_labels(sample):
    assert isinstance(sample, dict)
    sample['label'] = [1, 2, 3, 4]
    return sample


@pytest.mark.parametrize("data_list", ["test/resources/dataset/data.list"])
def test_dynamic_batch_datapipe(data_list):
    assert isinstance(data_list, str)
    epoch = 100
    dataset = WenetRawDatasetSource([data_list] * epoch)
    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(compute_fbank)
    dataset = dataset.map(fake_labels)
    dataset = dataset.map(partial(detect_language, limited_langs=['zh', 'en']))
    dataset = dataset.map(detect_task)
    max_frames_in_batch = 10000
    dataset = dataset.dynamic_batch(
        window_class=DynamicBatchWindow(max_frames_in_batch),
        wrapper_class=padding)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             num_workers=2)
    for d in dataloader:
        assert d['feats'].size(1) <= max_frames_in_batch


def test_bucket_batch_datapipe():
    dataset = datapipes.iter.IterableWrapper(range(10))

    def _seq_len_fn(elem):
        if elem < 5:
            return 2
        elif elem >= 5 and elem < 7:
            return 4
        else:
            return 8

    dataset = dataset.bucket_by_sequence_length(
        _seq_len_fn,
        bucket_boundaries=[3, 5],
        bucket_batch_sizes=[3, 2, 2],
    )
    expected = [
        [0, 1, 2],
        [5, 6],
        [7, 8],
        [3, 4],
        [9],
    ]
    result = []
    for d in dataset:
        result.append(d)
    assert len(result) == len(expected)
    for (r, h) in zip(expected, result):
        assert len(r) == len(h)
        assert all(rr == hh for (rr, hh) in zip(r, h))


def test_shuffle_deterministic():
    dataset = datapipes.iter.IterableWrapper(range(10))
    dataset = dataset.shuffle()

    generator = torch.Generator()
    generator.manual_seed(100)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             num_workers=0,
                                             generator=generator,
                                             persistent_workers=False)

    result = []
    for epoch in range(2):
        _ = epoch
        for d in dataloader:
            result.append(d)

    expected = [2, 7, 8, 9, 4, 6, 3, 0, 5, 1, 1, 6, 0, 5, 9, 8, 3, 2, 7, 4]
    for (r, h) in zip(result, expected):
        assert r == h


def _read_file(filename):
    if filename == 'b.txt':
        raise NotImplementedError('not found')
    return filename


def test_map_ignore_error_datapipe():
    file_list = ['a.txt', 'b.txt', 'c.txt']

    dataset = IterableWrapper(iter(file_list)).map_ignore_error(_read_file)
    result = []
    for d in dataset:
        result.append(d)
    expected = ['a.txt', 'c.txt']
    assert len(result) == len(expected)
    all(h == r for (h, r) in zip(result, expected))


def test_repeat():
    source = [1, 2, 3]
    epoch = 2

    dataset = IterableWrapper(source)
    dataset = RepeatDatapipe(dataset, epoch)
    expected = [1, 2, 3] * epoch

    result = []
    for elem in dataset:
        result.append(elem)
    assert len(result) == len(expected)
    assert all(h == r for (h, r) in zip(result, expected))

    source = [{"1.wav": "we"}, {"2.wav": "net"}, {"3.wav": "better"}]
    expected = [[{
        "1.wav": "we"
    }, {
        "2.wav": "net"
    }], [{
        "3.wav": "better"
    }, {
        "1.wav": "we"
    }], [{
        "2.wav": "net"
    }, {
        "3.wav": "better"
    }]]
    dataset = IterableWrapper(source)
    dataset = RepeatDatapipe(dataset, epoch).batch(2)
    result = []
    for elem in dataset:
        result.append(elem)

    assert len(result) == len(expected)
    all(h == r for h, r in zip(result, expected))


def test_interleave():
    dataset_1 = IterableWrapper(range(10))
    dataset_2 = IterableWrapper(range(20, 30, 2))

    dataset = InterlaveDataPipe([dataset_1, dataset_2])
    dataset = dataset.batch(4)
    generator = torch.Generator()
    generator.manual_seed(100)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=None,
                                             num_workers=0,
                                             generator=generator,
                                             persistent_workers=False)
    expected = [[0, 1, 2, 3], [4, 20, 5, 22], [24, 6, 7, 8], [26, 9, 28]]

    result = []
    for batch in dataloader:
        result.append(batch)

    assert expected == result
