from torch.utils.data import datapipes

from wenet.dataset.datapipes import SortDatPipes


def test_sort_datapipes():
    N = 10
    dataset = datapipes.iter.IterableWrapper(range(N))
    dataset = SortDatPipes(dataset, key_func=lambda elem: elem, reverse=True)
    for (i, d) in enumerate(dataset):
        assert d == N - 1 - i
