import io
import librosa
import random

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from wenet.text.base_tokenizer import BaseTokenizer

# torchaudio.utils.sox_utils.set_buffer_size(16500)


def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            data: str, str is a json line has key/wav/txt

        Returns:
            {key, wav, txt, sample_rate}
    """
    assert 'key' in sample
    assert 'wav' in sample
    assert 'txt' in sample
    wav_file = sample['wav']
    if 'start' in sample:
        assert 'end' in sample
        sample_rate = torchaudio.info(wav_file).sample_rate
        start_frame = int(sample['start'] * sample_rate)
        end_frame = int(sample['end'] * sample_rate)
        waveform, _ = torchaudio.load(filepath=wav_file,
                                      num_frames=end_frame - start_frame,
                                      frame_offset=start_frame)
    else:
        if isinstance(wav_file, str):
            with open(wav_file, 'rb') as f:
                wav_file = f.read()
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
        del sample['wav']
        sample['wav'] = waveform  # overwrite wav
        sample['sample_rate'] = sample_rate
    return sample


def resample(sample, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return random.sample


def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            Iterable{key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav

    return sample


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
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
    sample['feat'] = mat
    return sample


def compute_mfcc(sample,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: {key, wav, label, sample_rate}

        Returns:
            {key, feat, label}
    """
    assert 'wav' in sample
    assert 'key' in sample
    assert 'label' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.mfcc(waveform,
                     num_mel_bins=num_mel_bins,
                     frame_length=frame_length,
                     frame_shift=frame_shift,
                     dither=dither,
                     num_ceps=num_ceps,
                     high_freq=high_freq,
                     low_freq=low_freq,
                     sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def compute_log_mel_spectrogram(sample,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            data: {key, wav, label, sample_rate}

        Returns:
            {key, feat, label}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    assert 'label' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
    if padding > 0:
        waveform = F.pad(waveform, (0, padding))
    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    sample['feat'] = log_spec.transpose(0, 1)
    yield sample


def tokenize(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: {key, wav, txt, sample_rate}

        Returns:
            {key, wav, txt, tokens, label, sample_rate}
    """
    assert 'txt' in sample
    tokens, label = tokenizer.tokenize(sample['txt'])
    sample['tokens'] = tokens
    sample['label'] = label
    yield sample


def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
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
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, label}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


def spec_trim(sample, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    max_frames = x.size(0)
    length = random.randint(1, max_t)
    if length < max_frames / 2:
        y = x.clone().detach()[:max_frames - length]
        sample['feat'] = y
    yield sample
