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
import tarfile

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
    """
    for sample in data:
        assert 'url' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['url']
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
            file_obj = stream.extractfile(tarinfo)
            if postfix == 'txt':
                data['txt'] = file_obj.read().decode('utf8').strip()
            elif postfix in AUDIO_FORMAT_SETS:
                waveform, sample_rate = torchaudio.load(file_obj)
                data['wav'] = waveform
                data['sample_rate'] = sample_rate
            else:
                data[postfix] = file_ojb.read()
            prev_prefix = prefix
        if prev_prefix is not None:
            data['key'] = prev_prefix
            yield data
        stream.close()


def filter(data,
           max_length=10240,
           min_length=0,
           token_max_length=200,
           token_min_length=1):
    """ Filter sample according to feature and label length

        Attributes::
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def decode_text(data, symbol_table, bpe_model=None):
    """ Decode text to chars or BPE
    """
    # TODO(Binbin Zhang): Support BPE
    for sample in data:
        assert 'txt' in sample
        txt = sample['txt']
        label = []
        tokens = []
        for ch in txt:
            tokens.append(ch)
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])
        sample['tokens'] = tokens
        sample['label'] = label
        yield sample


def spec_augmentation(data,
                      num_t_mask=2,
                      num_f_mask=2,
                      max_t=50,
                      max_f=10,
                      max_w=80):
    """ Do spec augmentation

    Args:
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample
