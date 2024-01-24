import collections
from subprocess import PIPE, Popen
import tarfile
from urllib.parse import urlparse
from tensorboardX.writer import logging
from torch.utils.data import IterDataPipe, functional_datapipe
from torch.utils.data import datapipes
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


class TarParseError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


class UrlOpenError(TarParseError):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(msg, *args)


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


@functional_datapipe("ignore_error")
class IgnoreError(IterDataPipe):

    def __init__(self, datapipe: IterDataPipe, log: bool = True) -> None:
        super().__init__()
        self.dp = datapipe
        self.iter_dp = None
        self.log = log

    def __iter__(self):
        if self.iter_dp is None:
            self.iter_dp = iter(self.dp)
        while True:
            try:
                elem = next(self.iter_dp)
                yield elem
            except StopIteration as ex:
                self.iter_dp = None
                return
            except Exception as ex:
                if self.log:
                    logging.warning(str(ex))
                continue


@functional_datapipe('url_opener')
class UrlOpenPipe(IterDataPipe):

    def __init__(self, datapipe: IterDataPipe) -> None:
        super().__init__()
        self.dp = datapipe

    def __iter__(self):
        for elem in self.dp:
            assert 'file_name' in elem
            assert 'line' in elem
            try:
                assert isinstance(elem, dict)
                url = elem['line']
                pr = urlparse(url)
                # local file
                if pr.scheme == '' or pr.scheme == 'file':
                    stream = open(url, 'rb')
                    # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
                else:
                    cmd = f'wget -q -O - {url}'
                    process = Popen(cmd, shell=True, stdout=PIPE)
                    elem.update(process=process)
                    stream = process.stdout
                elem.update(stream=stream)
                yield elem
                del elem
            except Exception as ex:
                raise UrlOpenError(str(ex)) from ex


@functional_datapipe("tar_file_and_group")
class TarsDataPipe(IterDataPipe):
    """ Decode wenet's tar , yield {'txt': "...", "raw": "..."}
    """

    def __init__(self, datapipe: IterDataPipe) -> None:
        super().__init__()
        self.dp = datapipe

    def __iter__(self):
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
                                raise TarParseError(name + str(ex)) from ex
                            prev_prefix = prefix
                    if prev_prefix is not None:
                        example['key'] = prev_prefix
                        yield example
            except Exception as ex:
                msg = 'In tar_file_and_group: {} when processing '.format(ex)
                raise TarParseError(msg) from ex
            finally:
                if 'process' in sample:
                    sample['process'].communicate()
                sample['stream'].close()


class WenetRawDatasetSource(IterDataPipe):

    def __init__(self, filenames: str, prefetch: int = 500) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames).prefetch(
            prefetch).sharding_filter()

    def __iter__(self):
        for d in self.dp:
            yield d


class WenetTarShardDatasetSource(IterDataPipe):

    def __init__(self, filenames: str, prefetch: int = 500) -> None:
        super().__init__()
        self.dp = TextLineDataPipe(filenames).sharding_filter().url_opener(
        ).ignore_error(log=True).tar_file_and_group().ignore_error(
            log=True).prefetch(prefetch)

    def __iter__(self):
        for d in self.dp:
            yield d
