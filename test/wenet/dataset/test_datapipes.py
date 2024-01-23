from functools import partial
import torch
from torch.utils.data import datapipes

from wenet.dataset.datapipes import SortDatPipes


def test_sort_datapipe():
    N = 10
    dataset = datapipes.iter.IterableWrapper(range(N))
    dataset = SortDatPipes(dataset, key_func=lambda elem: elem, reverse=True)
    for (i, d) in enumerate(dataset):
        assert d == N - 1 - i


def accumate(size, max_len=4):
    accumate.n_feats += size
    if accumate.n_feats > max_len:
        accumate.n_feats = size
        return True
    else:
        return False


def padding(data):
    return torch.tensor(data)


def test_dynamic_datapipe():
    N = 10

    accumate.n_feats = 0
    window_fn = partial(accumate, max_len=10)
    dataset = datapipes.iter.IterableWrapper(range(N))
    dataset = dataset.dynamic_batch(window_fn, padding)
    expected = [
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([5]),
        torch.tensor([6]),
        torch.tensor([7]),
        torch.tensor([8]),
        torch.tensor([9])
    ]
    result = []
    for d in dataset:
        result.append(d)
    assert len(result) == len(expected)
    assert all(torch.allclose(r, e) for r, e in zip(result, expected))
