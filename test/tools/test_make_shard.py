import glob
import io
import torch
from torchaudio._extension import torchaudio


def test_save_load_consistently():
    wav_paths = glob.glob("test/resources/*.wav")
    for wav_path in wav_paths:
        wav, sr = torchaudio.load(wav_path)
        with io.BytesIO() as f:
            wav = torchaudio.transforms.Resample(sr, sr)(wav)
            wav_short = (wav * (1 << 15))
            wav_short = wav_short.to(torch.int16)
            torchaudio.save(f, wav_short, sr, format="wav", bits_per_sample=16)
            f.seek(0)
            b = f.read()

        with io.BytesIO(b) as f:
            new_wav, new_sr = torchaudio.load(f)
            assert new_sr == sr
            torch.allclose(new_wav, wav)


def test_sox_set_buffer():
    torchaudio.utils.sox_utils.set_buffer_size(16500)


def test_make_shards():
    # TODO(MDdct): add make shards
    pass
