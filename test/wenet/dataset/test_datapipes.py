import pytest
from torch.utils.data import datapipes

from wenet.dataset.datapipes import SortDataPipe, WenetRawDatasetSource
from wenet.dataset.dataset_v2 import fake_labels
from wenet.dataset.processor_v2 import (decode_wav, dynamic_batch_window_fn,
                                        padding, parse_json, compute_fbank)


def key_func(elem):
    return elem


def test_sort_datapipe():
    N = 10
    dataset = datapipes.iter.IterableWrapper(range(N))
    dataset = SortDataPipe(dataset, key_func=key_func, reverse=True)
    for (i, d) in enumerate(dataset):
        assert d == N - 1 - i


@pytest.mark.parametrize("data_list", ["test/resources/dataset/data.list"])
def test_dynamic_batch_datapipe(data_list):
    assert isinstance(data_list, str)
    epoch = 100
    dataset = WenetRawDatasetSource([data_list] * epoch)
    dataset = dataset.map(parse_json)
    dataset = dataset.map(decode_wav)
    dataset = dataset.map(compute_fbank)
    dataset = dataset.map(fake_labels)
    max_frames_in_batch = 10000
    dataset = dataset.dynamic_batch(
        dynamic_batch_window_fn(max_frames_in_batch=max_frames_in_batch),
        padding)

    for d in dataset:
        assert d['feats'].size(1) <= max_frames_in_batch
