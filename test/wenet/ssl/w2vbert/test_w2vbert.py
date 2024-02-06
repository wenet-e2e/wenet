from pathlib import Path
import pytest
import torch
import torchaudio

from wenet.dataset import processor

try:
    import fairseq2  # noqa
    from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
    from fairseq2.memory import MemoryBlock
except ImportError:
    import os
    os.system('pip install --no-input fairseq2')
    import fairseq2  # noqa
    from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
    from fairseq2.memory import MemoryBlock


@pytest.mark.parametrize(
    "wav_file",
    [
        # "test/resources/aishell-BAC009S0724W0121.wav",
        "test/resources/librispeech-1995-1837-0001.wav",
    ])
def test_w2vbert_fbank(wav_file):
    fbank_convert = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
    )
    audio_decoder = AudioDecoder(dtype=torch.float32)
    with Path(wav_file).open("rb") as fb:
        block = MemoryBlock(fb.read())
    decode_audio = audio_decoder(block)
    w2vbert_waveform = decode_audio['waveform']
    w2vbert_mat = fbank_convert(decode_audio)['fbank']

    wenet_waveform, _ = torchaudio.load(wav_file)
    fbank_args = {
        "num_mel_bins": 80,
        "frame_length": 25,
        "frame_shift": 10,
        "dither": 0.0,
    }
    sample = {'sample_rate': 16000, "wav": wenet_waveform, 'key': wav_file}
    wenet_mat = processor.compute_w2vbert_fbank(sample, **fbank_args)['feat']
    assert torch.allclose(w2vbert_waveform.transpose(0, 1), wenet_waveform)
    assert torch.allclose(w2vbert_mat, wenet_mat, atol=9e-5, rtol=9e-4)
