#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-21] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import pytest
import numpy as np
import torchaudio
import whisper

from wenet.dataset.processor import compute_log_mel_spectrogram
from wenet.transformer.embedding import WhisperPositionalEncoding


@pytest.mark.parametrize(
    "audio_path",
    [
        "test/resources/aishell-BAC009S0724W0121.wav",
        "test/resources/librispeech-1995-1837-0001.wav"
    ]
)
def test_load_audio(audio_path):
    waveform_wenet, sample_rate = torchaudio.load(audio_path)
    waveform_wenet = waveform_wenet.numpy().flatten().astype(np.float32)
    wavform_whisper = whisper.load_audio(audio_path)
    np.testing.assert_allclose(waveform_wenet, wavform_whisper,
                               rtol=1e-07, atol=1e-06)


@pytest.mark.parametrize(
    "audio_path",
    [
        "test/resources/aishell-BAC009S0724W0121.wav",
        "test/resources/librispeech-1995-1837-0001.wav"
    ]
)
def test_log_mel_spectrogram(audio_path):
    waveform_wenet, sample_rate = torchaudio.load(audio_path)
    sample = {"wav": waveform_wenet, "sample_rate": sample_rate,
              "key": audio_path, "label": "<N/A>"}
    log_spec_wenet = next(compute_log_mel_spectrogram([sample]))["feat"]
    log_spec_wenet = log_spec_wenet.transpose(0, 1).numpy().astype(np.float32)
    log_spec_whisper = whisper.log_mel_spectrogram(audio_path)
    np.testing.assert_allclose(log_spec_wenet, log_spec_whisper,
                               rtol=1e-07, atol=1e-06)


@pytest.mark.parametrize(
    "length,channels", [(512, 80), (1024, 128), (2048, 256), (4096, 512)],
)
def test_sinusoids(length, channels):
    sinusoids_whisper = whisper.model.sinusoids(length, channels, max_timescale=10000)
    sinusoids_wenet = WhisperPositionalEncoding(d_model=channels, dropout_rate=0.0,
                                                max_len=length)
    np.testing.assert_allclose(sinusoids_wenet.pe.squeeze(0).numpy(),
                               sinusoids_whisper.numpy(),
                               rtol=1e-08, atol=0)
