# Copyright (c) 2023 Wenet Community. (authors: Dinghao Zhou)
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

import collections
from collections.abc import Callable
import sys
import tarfile
import logging
from typing import List
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes
from torch.utils.data.datapipes.iter import Mapper
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES, ShardingFilterIterDataPipe)
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

from wenet.dataset.processor import parse_url


@functional_datapipe("map_ignore_error")
class MapperIgnoreErrorDataPipe(Mapper):

    def __init__(self,
                 dataset: IterDataPipe,
                 fn: Callable,
                 input_col=None,
                 output_col=None,
                 log_error: bool = True) -> None:
        super().__init__(dataset, fn, input_col, output_col)
        self._iter = None
        self.log_error = log_error

    def __iter__(self):
        if self._iter is None:
            self._iter = iter(self.datapipe)

        while True:
            try:
                elem = next(self._iter)
                yield self._apply_fn(elem)
            except StopIteration:
                self._iter = None
                return
            except Exception as ex:
                if self.log_error:
                    logging.warning(str(ex))


@functional_datapipe('bucket_by_sequence_length')
class BucketBySequenceLengthDataPipe(IterDataPipe):

    def __init__(
        self,
        dataset: IterDataPipe,
        elem_length_func,
        bucket_boundaries: List[int],
        bucket_batch_sizes: List[int],
        wrapper_class=None,
    ) -> None:
        super().__init__()
        _check_unpickable_fn(elem_length_func)
        assert len(bucket_batch_sizes) == len(bucket_boundaries) + 1
        self.bucket_batch_sizes = bucket_batch_sizes
        self.bucket_boundaries = bucket_boundaries + [sys.maxsize]
        self.elem_length_func = elem_length_func

        self._group_dp = GroupByWindowDataPipe(dataset,
                                               self._element_to_bucket_id,
                                               self._window_size_func,
                                               wrapper_class=wrapper_class)

    def __iter__(self):
        yield from self._group_dp

    def _element_to_bucket_id(self, elem):
        seq_len = self.elem_length_func(elem)
        bucket_id = 0
        for (i, b) in enumerate(self.bucket_boundaries):
            if seq_len < b:
                bucket_id = i
                break
        return bucket_id

    def _window_size_func(self, bucket_id):
        return self.bucket_batch_sizes[bucket_id]


@functional_datapipe("group_by_window")
class GroupByWindowDataPipe(datapipes.iter.Grouper):

    def __init__(
        self,
        dataset: IterDataPipe,
        key_func,
        window_size_func,
        wrapper_class=None,
    ):
        super().__init__(dataset,
                         key_func,
                         keep_key=False,
                         group_size=None,
                         drop_remaining=False)
        _check_unpickable_fn(window_size_func)
        self.dp = dataset
        self.window_size_func = window_size_func
        if wrapper_class is not None:
            _check_unpickable_fn(wrapper_class)
            del self.wrapper_class
            self.wrapper_class = wrapper_class

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            group_size = self.window_size_func(key)
            if group_size == len(self.buffer_elements[key]):
                result = self.wrapper_class(self.buffer_elements[key])
                yield result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield result


@functional_datapipe("sort")
class SortDataPipe(IterDataPipe):

    def __init__(self,
                 dataset: IterDataPipe,
                 buffer_size: int = 500,
                 key_func=None,
                 reverse=False) -> None:
        if key_func is not None:
            _check_unpickable_fn(key_func)
        self.buffer_size = buffer_size
        super().__init__()
        self.dp = dataset
        self._buffer = []
        self.key_func = key_func
        self.reverse = reverse

    def __iter__(self):
        for elem in self.dp:
            self._buffer.append(elem)
            if len(self._buffer) >= self.buffer_size:
                self._buffer.sort(key=self.key_func, reverse=self.reverse)
                for x in self._buffer:
                    yield x
                del self._buffer
                self._buffer = []
        # The sample left over
        self._buffer.sort(key=self.key_func, reverse=self.reverse)
        for x in self._buffer:
            yield x
        del self._buffer
        self._buffer = []


@functional_datapipe("dynamic_batch")
class DynamicBatchDataPipe(IterDataPipe):

    def __init__(self, dataset: IterDataPipe, window_class,
                 wrapper_class) -> None:
        _check_unpickable_fn(window_class)
        _check_unpickable_fn(wrapper_class)
        super().__init__()
        self.dp = dataset
        assert window_class is not None
        assert wrapper_class is not None
        self.window_class = window_class
        self._buffer = []
        self._wrappr_class = wrapper_class

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, len(self._buffer)):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._wrappr_class(self._buffer)
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._wrappr_class(self._buffer)
        del self._buffer
        self._buffer = []


@functional_datapipe("prefetch")
class PrefetchDataPipe(IterDataPipe):
    """Performs prefetching"""

    def __init__(
        self,
        dataset: IterDataPipe,
        buffer_size: int = 500,
    ):
        # TODO(Mddct): support multiprocessing pool with shared-memory to
        #   prefetch
        super().__init__()
        self.dp = dataset
        self._iter = None
        self._prefetch_buffer_size = buffer_size
        self._buffer = None
        if self._prefetch_buffer_size > 0:
            self._buffer = collections.deque(maxlen=self._prefetch_buffer_size)

    def __iter__(self):
        if self._prefetch_buffer_size > 0:
            if self._iter is None:
                self._iter = iter(self.dp)
            assert self._buffer is not None

            while True:
                if len(self._buffer) <= self._prefetch_buffer_size // 2:
                    while len(self._buffer) < self._prefetch_buffer_size:
                        try:
                            self._buffer.append(next(self._iter))
                        except StopIteration:
                            if len(self._buffer) != 0:
                                while len(self._buffer) > 0:
                                    yield self._buffer.popleft()
                            self._iter = None
                            return
                while len(self._buffer) > self._prefetch_buffer_size // 2:
                    elem = self._buffer.popleft()
                    yield elem

        else:
            yield from self.dp


@functional_datapipe("shard")
class ShardDataPipe(ShardingFilterIterDataPipe):

    def __init__(self, dataset: IterDataPipe, partition: bool = False):
        super().__init__(dataset, None)
        self.partition = partition
        self.dp = dataset

    def apply_sharding(self, num_of_instances: int, instance_id: int,
                       sharding_group: SHARDING_PRIORITIES):
        if self.partition:
            return super().apply_sharding(num_of_instances, instance_id,
                                          sharding_group)
        else:
            # We can not handle uneven data for CV on DDP, so we don't
            # sample data by rank, that means every GPU gets the same
            # and all the CV data
            info = torch.utils.data.get_worker_info()
            if info is None:
                self.num_of_instances = 1
                self.instance_id = 0
            else:
                n_workers_per_device = info.num_workers
                self.num_of_instances = n_workers_per_device
                self.instance_id = info.id


class TextLineDataPipe(IterDataPipe):
    """ Streamming Text line
    """

    def __init__(self, filenames, mode='r'):
        super().__init__()
        _dp = datapipes.iter.FileLister(filenames)
        _dp = datapipes.iter.FileOpener(_dp, mode=mode)
        self.dp = _dp

    def __iter__(self):
        for fname, stream in self.dp:
            for line in stream:
                line = line.strip('\n')
                yield {"file_name": fname, "line": line}
            stream.close()


@functional_datapipe("tar_file_and_group")
class TarsDataPipe(IterDataPipe):
    """ Decode wenet's tar , yield {'txt': "...", "raw": "..."}
    """

    def __init__(self, dataset: IterDataPipe) -> None:
        super().__init__()
        self.dp = dataset

    def __iter__(self):
        from wenet.dataset.processor import AUDIO_FORMAT_SETS
        for sample in self.dp:
            assert 'file_name' in sample
            assert 'line' in sample
            assert 'stream' in sample
            try:
                with tarfile.open(fileobj=sample['stream'],
                                  mode="r:*") as stream:
                    prev_prefix = None
                    example = {
                        'file_name': sample['file_name'],
                        'tar_file_name': sample['line']
                    }
                    valid = True
                    for tarinfo in stream:
                        name = tarinfo.name
                        pos = name.rfind('.')
                        assert pos > 0
                        prefix, postfix = name[:pos], name[pos + 1:]
                        if prev_prefix is not None and prefix != prev_prefix:
                            example['key'] = prev_prefix
                            if valid:
                                yield example
                            example = {
                                'file_name': sample['file_name'],
                                'tar_file_name': sample['line']
                            }
                            valid = True
                        with stream.extractfile(tarinfo) as file_obj:
                            try:
                                if postfix == 'txt':
                                    example['txt'] = file_obj.read().decode(
                                        'utf8').strip()
                                elif postfix in AUDIO_FORMAT_SETS:
                                    example['wav'] = file_obj.read()
                                else:
                                    example[postfix] = file_obj.read()
                            except Exception as ex:
                                valid = False
                                logging.warning(
                                    'error to parse {}'.format(name))
                            prev_prefix = prefix
                    if prev_prefix is not None:
                        example['key'] = prev_prefix
                        yield example
            except Exception as ex:
                msg = 'In tar_file_and_group: {} when processing {}'.format(
                    ex, sample['line'])
                logging.warning(msg)
            finally:
                if 'process' in sample:
                    sample['process'].communicate()
                sample['stream'].close()


class WenetRawDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition=True) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames).prefetch(prefetch).shard(
            partition)

    def __iter__(self):
        for d in self.dp:
            yield d


class WenetTarShardDatasetSource(IterDataPipe):

    def __init__(self,
                 filenames: str,
                 prefetch: int = 500,
                 partition: bool = False) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames).shard(
            partition).map_ignore_error(
                parse_url).tar_file_and_group().prefetch(prefetch)

    def __iter__(self):
        for d in self.dp:
            yield d
