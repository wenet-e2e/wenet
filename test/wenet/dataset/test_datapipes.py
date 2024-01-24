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


# def accumate(size, max_len=4):
#     accumate.n_feats += size
#     if accumate.n_feats > max_len:
#         accumate.n_feats = size
#         return True
#     else:
#         return False

# def padding(data):
#     return torch.tensor(data)

# def test_dynamic_batch_datapipe():
#     N = 10

#     accumate.n_feats = 0
#     window_fn = partial(accumate, max_len=10)
#     dataset = datapipes.iter.IterableWrapper(range(N))
#     dataset = dataset.dynamic_batch(window_fn, padding)
#     expected = [
#         torch.tensor([0, 1, 2, 3, 4]),
#         torch.tensor([5]),
#         torch.tensor([6]),
#         torch.tensor([7]),
#         torch.tensor([8]),
#         torch.tensor([9])
#     ]
#     result = []
#     for d in dataset:
#         result.append(d)
#     assert len(result) == len(expected)
#     assert all(torch.allclose(r, e) for r, e in zip(result, expected))


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
        assert d['feats'].size(0) <= max_frames_in_batch
