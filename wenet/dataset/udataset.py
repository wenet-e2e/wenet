# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

import wenet.dataset.processor as processor
import wenet.dataset.utils as utils


class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:
    def __init__(self, shuffle=False):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = data.copy()
        if self.shuffle:
            random.Random(self.epoch).shuffle(data)
        data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):
    def __init__(self, lists, shuffle=False):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle)

    def set_epoch(self):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        lists = self.sampler.sample(self.lists)
        for src in lists:
            # yield dict(src=src)
            data = dict(src=src)
            data.update(sampler_info)
            yield data


def Dataset(data_type, data_list_file, symbol_table_file):
    """ Construct dataset from arguments
        Args:
            data_type(str): raw/shard
    """
    assert data_type in ['raw', 'shard']
    lists = utils.read_lists(data_list_file)
    symbol_table = utils.read_symbol_table(symbol_table_file)
    dataset = DataList(lists)
    if dataset == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    else:
        dataset = Processor(dataset, processor.parse_raw)
    dataset = Processor(dataset, processor.decode_text, symbol_table)
    dataset = Processor(dataset, processor.filter)
    dataset = Processor(dataset, processor.resample)
    dataset = Processor(dataset, processor.compute_fbank)
    dataset = Processor(dataset, processor.spec_augmentation)
    dataset = Processor(dataset, processor.shuffle, 1000)
    dataset = Processor(dataset, processor.sort, 1000)
    dataset = Processor(dataset, processor.static_batch, 2)
    dataset = Processor(dataset, processor.padding)
    return dataset
