# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
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

import tarfile

from torch.utils.data import IterableDataset


class Processor(IterableDataset):
    def __init__(self, source, f):
        assert callable(f)
        self.source = source
        self.f = f

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source))

    def apply(self, f):
        assert callable(f)
        return Processor(self, f)


class ShardList(IterableDataset):
    def __init__(self, urls):
        self.urls = urls

    def set_epoch(self, epoch: int):
        pass

    def __iter__(self):
        for url in self.urls:
            yield dict(url=url)


def url_opener(data):
    """ Give url or local file, return file descriptor
    """
    for sample in data:
        assert 'url' in sample
        url = sample['url']
        print(url)
        stream = open(url, 'rb')
        sample.update(stream=stream)
        yield sample


def tar_file_and_group(data):
    """Expand a stream of open tar files into a stream of tar file contents.
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        data = {}
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                data['key'] = prev_prefix
                yield data
                data = {}
            data[postfix] = stream.extractfile(tarinfo).read()
            prev_prefix = prefix
        if prev_prefix is not None:
            data['key'] = prev_prefix
            yield data


def ShardDataset(urls):
    shards = ShardList(urls)
    shards = Processor(shards, url_opener)
    shards = Processor(shards, tar_file_and_group)
    return shards


if __name__ == '__main__':
    urls = ['sample.tgz']
    dataset = ShardDataset(urls)
    for item in dataset:
        print(item)
