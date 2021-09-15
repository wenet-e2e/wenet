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

from torch.utils.data import IterableDataset

import processor
import utils


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


class ShardList(IterableDataset):
    def __init__(self, urls):
        self.urls = urls

    def set_epoch(self, epoch: int):
        pass

    def __iter__(self):
        for url in self.urls:
            yield dict(url=url)


def ShardDataset(urls, symbol_table):
    dataset = ShardList(urls)
    dataset = Processor(dataset, processor.url_opener)
    dataset = Processor(dataset, processor.tar_file_and_group)
    dataset = Processor(dataset, processor.decode_text, symbol_table)
    dataset = Processor(dataset, processor.filter)
    dataset = Processor(dataset, processor.resample)
    dataset = Processor(dataset, processor.compute_fbank)
    dataset = Processor(dataset, processor.spec_augmentation)
    return dataset


if __name__ == '__main__':
    shard_list = '/export/maryland/binbinzhang/code/wenet/examples/aishell/s0/shards/train.list'
    symbol_table_file = '/export/maryland/binbinzhang/code/wenet/examples/aishell/s0/data/dict/lang_char.txt'
    urls = utils.read_urls_list(shard_list)
    symbol_table = utils.read_symbol_table(symbol_table_file)
    dataset = ShardDataset(urls, symbol_table)
    count = 0
    for item in dataset:
        print(item)
        count += 1
        if count > 1:
            break
