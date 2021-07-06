# coding: utf-8
# Author：WangTianRui
# Date ：2021/5/15 22:09
import torchaudio, os, random
import librosa as lib
import soundfile as sf
import torch.nn.functional as torchF
import torchaudio.compliance.kaldi as kaldi
import math
import torchaudio.sox_effects as sox_effects
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import get_window

EPS = torch.tensor(torch.finfo(torch.float).eps)


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    elif win_type == "povey":
        window = torch.hann_window(win_len, periodic=False).pow(0.85).data.numpy()
    else:
        window = get_window(win_type, win_len, fftbins=True)  # win_len

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T  # 514,400

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = np.int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = torchF.pad(inputs, [(self.win_len - self.stride), (self.win_len - self.stride)])
        outputs = torchF.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs  # B,F,T
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real ** 2 + imag ** 2)
            phase = torch.atan2(imag, real)
            return mags, phase


def load_wav(path, speed, dither, to_torch=True, VAD=False):
    assert os.path.exists(path), "wav path error"
    if speed == 1.0:
        wav, sr = sf.read(path, dtype="float32")
        if VAD:
            _, index = lib.effects.trim(wav, top_db=60)
    else:
        # 语速调整
        wav_info = sf.info(path)
        ta_no = torchaudio.__version__.split(".")
        ta_version = 100 * int(ta_no[0]) + 10 * int(ta_no[1])
        if ta_version < 80:
            # Note: deprecated in torchaudio>=0.8.0
            E = sox_effects.SoxEffectsChain()
            E.append_effect_to_chain('speed', speed)
            E.append_effect_to_chain("rate", wav_info.samplerate)
            E.set_input_file(path)
            wav, sr = E.sox_build_flow_effects()
        else:
            # Note: enable in torchaudio>=0.8.0
            wav, sr = sox_effects.apply_effects_file(path,
                                                     [['speed', str(speed)],
                                                      ['rate', str(wav_info.samplerate)]])
    wav = wav * (1 << 15)
    if to_torch:
        wav = torch.tensor(wav)
    # print(wav)
    if dither != 0.0:
        x = torch.max(EPS, torch.rand(wav.shape, device=wav.device, dtype=wav.dtype))
        rand_gauss = torch.sqrt(-2 * x.log()) * torch.cos(2 * math.pi * x)
        wav = wav + rand_gauss * dither
    if VAD:
        return wav.flatten(), sr, index
    else:
        return wav.flatten(), sr


class Features(nn.Module):
    def __init__(self, win_len, win_hop, n_fft, win_type="hanning", fbank_conf=None):
        super(Features, self).__init__()
        self.win_len = win_len
        self.win_hop = win_hop
        self.n_fft = n_fft
        self.win_type = win_type
        self.fbank_conf = fbank_conf

        self.stft = ConvSTFT(win_len=win_len, win_inc=win_hop, win_type=win_type, fft_len=n_fft,
                             feature_type="complex")
        if fbank_conf is not None:
            self.register_buffer("mel_energies", self.get_mel_energies())
        self.register_buffer("EPS", torch.tensor(torch.finfo(torch.float).eps))

    def get_mel_energies(self):
        """
        获取mel映射谱。
        fbank_conf = {
            "n_bin" : 80,
            "sr":16000.0,
            "low_freq":20.0,
            "high_freq":0.0,
            "vtln_low":100.0,
            "vtln_high":-500.0,
            "vtln_wrap":1.0,
            "use_log_fbank":True,
            "preemphasis_coefficient":0.97
        }
        :return: 1,1,n_bins,win_len//2+1
        """
        mel_energies = kaldi.get_mel_banks(self.fbank_conf["n_bin"], window_length_padded=self.n_fft,
                                           sample_freq=self.fbank_conf["sr"],
                                           low_freq=self.fbank_conf["low_freq"],
                                           high_freq=self.fbank_conf["high_freq"],
                                           vtln_low=self.fbank_conf["vtln_low"],
                                           vtln_high=self.fbank_conf["vtln_high"],
                                           vtln_warp_factor=self.fbank_conf["vtln_wrap"])[0]
        mel_energies = torchF.pad(mel_energies, [0, 1], mode="constant", value=0.0).unsqueeze(0).unsqueeze(0)
        return mel_energies

    def mag(self, x):
        """
        :param x:B,2*F,T
        :return: B,F,T
        """
        return torch.stack(torch.chunk(x, 2, dim=-2), dim=-1).pow(2).sum(dim=-1)

    def mag2fbank(self, mag):
        """
        :param mag:B,F,T
        :return: B,T,n_bins
        """
        mag = mag.permute(0, 2, 1).contiguous().unsqueeze(-2)  # B,T,1,F
        fbank = (mag * self.mel_energies).sum(-1)
        if self.fbank_conf["use_log_fbank"]:
            fbank = torch.log(torch.max(fbank, self.EPS))
        return fbank

    def forward(self, signal):
        """
        :param signal:B,T
        :return: B,T,F
        """
        # print(signal.size())
        if self.fbank_conf is not None and self.fbank_conf["preemphasis_coefficient"] != 0.0:
            offset = torchF.pad(signal.unsqueeze(0), [1, 0], mode="replicate").squeeze(0)  # B,1+T
            signal = signal - self.fbank_conf["preemphasis_coefficient"] * offset[:, :-1]
        stft = self.stft(signal)  # B,2*F,T
        if self.fbank_conf is not None:
            return self.mag2fbank(self.mag(stft))
        else:
            return stft.permute(0, 2, 1).contiguous()


class SpecAug(nn.Module):
    def __init__(self, num_t_mask, num_f_mask, max_t, max_f, max_w):
        """
        :param num_t_mask: num of time mask
        :param num_f_mask:
        :param max_t: max width of time mask
        :param max_f:
        :param max_w:
        """
        super(SpecAug, self).__init__()
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f
        self.max_w = max_w

    def forward(self, spec=None, spec_clean=None, t=-1, f=-1):
        if spec is not None:
            t, f = spec.size()
            mask = torch.ones_like(spec)
            # time mask
            for i in range(self.num_t_mask):
                start = random.randint(0, t - 1)
                length = random.randint(1, self.max_t)
                end = min(t, start + length)
                mask[start:end, :] = 0.0
            # freq mask
            for i in range(self.num_f_mask):
                start = random.randint(0, f - 1)
                length = random.randint(1, self.max_f)
                end = min(self.max_f, start + length)
                mask[:, start:end] = 0.0
            if spec_clean is None:
                return spec * mask
            else:
                return spec * mask, spec_clean * mask
        else:
            assert t > 0 and f > 0, "T,F should bigger than 0 but T;F=" + str(t) + ";" + str(f)
            mask = torch.ones(t, f)
            # time mask
            for i in range(self.num_t_mask):
                start = random.randint(0, t - 1)
                length = random.randint(1, self.max_t)
                end = min(t, start + length)
                mask[start:end, :] = 0.0
            # freq mask
            for i in range(self.num_f_mask):
                start = random.randint(0, f - 1)
                length = random.randint(1, self.max_f)
                end = min(self.max_f, start + length)
                mask[:, start:end] = 0.0
            return mask


if __name__ == '__main__':
    clean_test_path = r"../wav_test/clean4778.wav"
    noise_test_path = r"../wav_test/noise4778.wav"
    noisy_test_path = r"../wav_test/noisy4778.wav"
    signal, sr = load_wav(clean_test_path, speed=1.0, dither=0.0, to_torch=False)
    signal = signal[:16000 * 16 - 160]
    # play_wav(signal, sr)
    win_len = 400
    stride = 160
    fbank_test = Features(win_len=win_len, win_hop=stride, n_fft=512, win_type="povey", fbank_conf={
        "n_bin": 80,
        "sr": 16000.0,
        "low_freq": 20.0,
        "high_freq": 0.0,
        "vtln_low": 100.0,
        "vtln_high": -500.0,
        "vtln_wrap": 1.0,
        "use_log_fbank": True,
        "preemphasis_coefficient": 0.97
    })
    fbank = fbank_test(torch.tensor([signal]))
    print(fbank.size())
    frames = int((len(signal) + 2 * (win_len - stride) - (win_len - 1) - 1) / stride + 1)
    print(frames)

    # drawer.plot_mesh(fbank[0].data)
    # fbank_kaidi = kaldi.fbank(torch.tensor([signal]), num_mel_bins=80, frame_length=512 / 16000.0 * 1000,
    #                           frame_shift=256 / 16000.0 * 1000, remove_dc_offset=False,
    #                           energy_floor=0.0, sample_frequency=16000, preemphasis_coefficient=0.97)
    # print(fbank_kaidi.size())
    # drawer.plot_mesh(fbank_kaidi.data)
    # drawer.plot_mesh(fbank_kaidi - fbank[0])
    # print(torch.mean(fbank_kaidi - fbank[0]))
