#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-21] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import pytest
import torch
import torchaudio
import whisper
import yaml

import numpy as np
import torch.nn.functional as F

from whisper.audio import N_FFT, HOP_LENGTH, N_SAMPLES, N_FRAMES, pad_or_trim

from wenet.dataset.processor import compute_log_mel_spectrogram
from wenet.text.whisper_tokenizer import WhisperTokenizer
from wenet.transformer.embedding import WhisperPositionalEncoding
from wenet.whisper.convert_whisper_to_wenet_config_and_ckpt import (
    convert_to_wenet_yaml, convert_to_wenet_state_dict, convert_to_wenet_units)
from wenet.utils.common import add_whisper_tokens
from wenet.utils.init_model import init_model
from wenet.utils.mask import make_pad_mask, subsequent_mask

torch.manual_seed(777)
np.random.seed(777)


class DummyArguments:
    jit = False
    enc_init = None
    checkpoint = None


@pytest.mark.parametrize("audio_path", [
    "test/resources/aishell-BAC009S0724W0121.wav",
    "test/resources/librispeech-1995-1837-0001.wav"
])
def test_load_audio(audio_path):
    waveform_wenet, sample_rate = torchaudio.load(audio_path)
    waveform_wenet = waveform_wenet.numpy().flatten().astype(np.float32)
    wavform_whisper = whisper.load_audio(audio_path)
    np.testing.assert_allclose(waveform_wenet,
                               wavform_whisper,
                               rtol=1e-7,
                               atol=1e-10)


@pytest.mark.parametrize("audio_path", [
    "test/resources/aishell-BAC009S0724W0121.wav",
    "test/resources/librispeech-1995-1837-0001.wav"
])
def test_log_mel_spectrogram(audio_path):
    waveform_wenet, sample_rate = torchaudio.load(audio_path)
    sample = {
        "wav": waveform_wenet,
        "sample_rate": sample_rate,
        "key": audio_path,
        "label": "<N/A>"
    }
    log_spec_wenet = compute_log_mel_spectrogram(sample,
                                                 n_fft=N_FFT,
                                                 hop_length=HOP_LENGTH,
                                                 num_mel_bins=128,
                                                 padding=0)["feat"]
    log_spec_wenet = log_spec_wenet.transpose(0, 1).numpy().astype(np.float32)
    log_spec_whisper = whisper.log_mel_spectrogram(audio_path,
                                                   n_mels=128,
                                                   padding=0)
    np.testing.assert_allclose(log_spec_wenet,
                               log_spec_whisper,
                               rtol=1e-7,
                               atol=1e-10)


@pytest.mark.parametrize(
    "length,channels",
    [(512, 80), (1024, 128), (2048, 256), (4096, 512)],
)
def test_sinusoids(length, channels):
    sinusoids_whisper = whisper.model.sinusoids(length,
                                                channels,
                                                max_timescale=10000)
    sinusoids_wenet = WhisperPositionalEncoding(d_model=channels,
                                                dropout_rate=0.0,
                                                max_len=length)
    np.testing.assert_allclose(sinusoids_wenet.pe.squeeze(0).numpy(),
                               sinusoids_whisper.numpy(),
                               rtol=1e-7,
                               atol=1e-10)


@pytest.mark.parametrize("model,audio_path", [
    ("tiny", "test/resources/aishell-BAC009S0724W0121.wav"),
    ("base", "test/resources/librispeech-1995-1837-0001.wav"),
])
def test_model(model, audio_path):
    default = os.path.join(os.path.expanduser("~"), ".cache")
    download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default),
                                 "whisper", "{}".format(model))
    language = "zh"
    task = "transcribe"

    # 1. Init whisper
    whisper_model = whisper.load_model(model,
                                       device="cpu",
                                       download_root=download_root).float()
    whisper_model.eval()

    # 2. Init wenet
    checkpoint = torch.load("{}/{}.pt".format(download_root, model),
                            map_location="cpu")
    multilingual = checkpoint["dims"]['n_vocab'] >= 51865
    num_languages = checkpoint["dims"]['n_vocab'] - 51765 - int(multilingual)
    tokenizer = WhisperTokenizer(multilingual,
                                 num_languages=num_languages,
                                 language=language,
                                 task=task)
    tokenizer._build_tiktoken()

    convert_to_wenet_state_dict(
        checkpoint["model_state_dict"],
        os.path.join(download_root, 'wenet_whisper.pt'))
    convert_to_wenet_units(tokenizer.tokenizer,
                           os.path.join(download_root, 'units.txt'))
    convert_to_wenet_yaml(tokenizer.tokenizer, checkpoint["dims"],
                          os.path.join(download_root, 'train.yaml'))
    with open("{}/train.yaml".format(download_root), 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
        configs['cmvn_file'] = None
    args = DummyArguments()
    args.checkpoint = "{}/wenet_whisper.pt".format(download_root)
    wenet_model, _ = init_model(args, configs)
    torch.jit.script(wenet_model)
    wenet_model.eval()

    with torch.no_grad():
        _, dummy_tokens = tokenizer.tokenize("WeNet x OpenAI")

        # 3. Forward whisper.encoder
        mel1 = whisper.log_mel_spectrogram(audio_path,
                                           whisper_model.dims.n_mels,
                                           padding=N_SAMPLES).unsqueeze(0)
        whisper_mel = pad_or_trim(mel1, N_FRAMES, axis=-1)
        x = F.gelu(whisper_model.encoder.conv1(whisper_mel))
        x = F.gelu(whisper_model.encoder.conv2(x))
        whisper_subed = x.permute(0, 2, 1)
        whisper_embed = whisper_subed + whisper_model.encoder.positional_embedding
        x = whisper_embed
        whisper_layers_ouput = []
        for i, layer in enumerate(whisper_model.encoder.blocks):
            prev_x = x.clone()
            attn_ln_x = layer.attn_ln(x)
            whisper_layers_ouput.append({
                "name": "enc.layer{}.attn_ln".format(i),
                "value": attn_ln_x.clone()
            })
            attn_x = layer.attn(attn_ln_x, mask=None, kv_cache=None)[0]
            whisper_layers_ouput.append({
                "name": "enc.layer{}.attn".format(i),
                "value": attn_x.clone()
            })
            x = x + attn_x
            whisper_layers_ouput.append({
                "name":
                "enc.layer{}.attn_residul".format(i),
                "value":
                x.clone()
            })
            mlp_ln_x = layer.mlp_ln(x)
            whisper_layers_ouput.append({
                "name": "enc.layer{}.mlp_ln".format(i),
                "value": mlp_ln_x.clone()
            })
            mlp_x = layer.mlp(mlp_ln_x)
            whisper_layers_ouput.append({
                "name": "enc.layer{}.mlp".format(i),
                "value": mlp_x.clone()
            })
            x = x + mlp_x
            whisper_layers_ouput.append({
                "name":
                "enc.layer{}.mlp_residul".format(i),
                "value":
                x.clone()
            })
            np.testing.assert_allclose(x.numpy(),
                                       layer(prev_x).numpy(),
                                       rtol=1e-7,
                                       atol=1e-10)
        whisper_encoder_out = whisper_model.encoder.ln_post(x)
        np.testing.assert_allclose(whisper_encoder_out.numpy(),
                                   whisper_model.encoder(whisper_mel).numpy(),
                                   rtol=1e-7,
                                   atol=1e-10)

        # 4. Forward whisper.decoder
        whisper_tokens = torch.tensor(
            list(tokenizer.tokenizer.sot_sequence) +
            [tokenizer.tokenizer.no_timestamps] + dummy_tokens,
            dtype=torch.long).unsqueeze(0)  # (B=1, 9)
        whisper_decoder_embed = whisper_model.decoder.token_embedding(
            whisper_tokens)
        pos_func = whisper_model.decoder.positional_embedding
        whisper_decoder_pos = pos_func[:whisper_decoder_embed.
                                       shape[1], :].unsqueeze(0)
        whisper_decoder_embed_posed = whisper_decoder_embed + whisper_decoder_pos
        x = whisper_decoder_embed_posed.clone()
        for i, layer in enumerate(whisper_model.decoder.blocks):
            prev_x = x.clone()
            attn_ln_x = layer.attn_ln(x)
            whisper_layers_ouput.append({
                "name": "dec.layer{}.attn_ln".format(i),
                "value": attn_ln_x.clone()
            })
            attn_x = layer.attn(attn_ln_x,
                                mask=whisper_model.decoder.mask,
                                kv_cache=None)[0]
            whisper_layers_ouput.append({
                "name": "dec.layer{}.attn".format(i),
                "value": attn_x.clone()
            })
            x = x + attn_x
            whisper_layers_ouput.append({
                "name":
                "dec.layer{}.attn_residul".format(i),
                "value":
                x.clone()
            })
            cross_attn_ln_x = layer.cross_attn_ln(x)
            whisper_layers_ouput.append({
                "name":
                "dec.layer{}.cross_attn_ln".format(i),
                "value":
                cross_attn_ln_x.clone()
            })
            cross_attn_x = layer.cross_attn(cross_attn_ln_x,
                                            whisper_encoder_out,
                                            mask=None,
                                            kv_cache=None)[0]
            whisper_layers_ouput.append({
                "name":
                "dec.layer{}.cross_attn".format(i),
                "value":
                cross_attn_x.clone()
            })
            x = x + cross_attn_x
            whisper_layers_ouput.append({
                "name": f"dec.layer{i}.cross_attn_residul",
                "value": x.clone()
            })
            mlp_ln_x = layer.mlp_ln(x)
            whisper_layers_ouput.append({
                "name": "dec.layer{}.mlp_ln".format(i),
                "value": mlp_ln_x.clone()
            })
            mlp_x = layer.mlp(mlp_ln_x)
            whisper_layers_ouput.append({
                "name": "dec.layer{}.mlp".format(i),
                "value": mlp_x.clone()
            })
            x = x + mlp_x
            whisper_layers_ouput.append({
                "name":
                "dec.layer{}.mlp_residul".format(i),
                "value":
                x.clone()
            })
            np.testing.assert_allclose(x.numpy(),
                                       layer(prev_x,
                                             whisper_encoder_out,
                                             mask=whisper_model.decoder.mask,
                                             kv_cache=None).numpy(),
                                       rtol=1e-7,
                                       atol=1e-10)
        x = whisper_model.decoder.ln(x)
        whisper_logits = (x @ torch.transpose(
            whisper_model.decoder.token_embedding.weight, 0, 1))
        np.testing.assert_allclose(whisper_logits.numpy(),
                                   whisper_model.decoder(
                                       whisper_tokens,
                                       whisper_encoder_out).numpy(),
                                   rtol=1e-7,
                                   atol=1e-10)

        # 5. Forward wenet.encoder
        waveform, sample_rate = torchaudio.load(audio_path)
        sample = {
            "wav": waveform,
            "sample_rate": sample_rate,
            "key": audio_path,
            "label": "<N/A>"
        }
        mel2 = compute_log_mel_spectrogram(
            sample,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            num_mel_bins=whisper_model.dims.n_mels,
            padding=N_SAMPLES)["feat"].unsqueeze(0)
        wenet_mel = pad_or_trim(mel2, N_FRAMES, axis=-2)
        T = wenet_mel.size(1)
        masks = ~make_pad_mask(torch.tensor([T], dtype=torch.long),
                               T).unsqueeze(1)  # (B=1, 1, T)
        wenet_embed, pos_emb, masks = wenet_model.encoder.embed(
            wenet_mel, masks)
        wenet_subed = wenet_embed - pos_emb
        x = wenet_embed
        wenet_layers_output = []
        for i, layer in enumerate(wenet_model.encoder.encoders):
            prev_x = x
            attn_ln_x = layer.norm1(x)
            wenet_layers_output.append({
                "name": "enc.layer{}.attn_ln".format(i),
                "value": attn_ln_x.clone()
            })
            x_att, _ = layer.self_attn(attn_ln_x,
                                       attn_ln_x,
                                       attn_ln_x,
                                       masks,
                                       cache=(torch.zeros((0, 0, 0, 0)),
                                              torch.zeros(0, 0, 0, 0)))
            wenet_layers_output.append({
                "name": "enc.layer{}.attn".format(i),
                "value": x_att.clone()
            })
            x = x + x_att
            wenet_layers_output.append({
                "name":
                "enc.layer{}.attn_residul".format(i),
                "value":
                x.clone()
            })

            mlp_ln_x = layer.norm2(x)
            wenet_layers_output.append({
                "name": "enc.layer{}.mlp_ln".format(i),
                "value": mlp_ln_x.clone()
            })
            mlp_x = layer.feed_forward(mlp_ln_x)
            wenet_layers_output.append({
                "name": "enc.layer{}.mlp".format(i),
                "value": mlp_x.clone()
            })
            x = x + mlp_x
            wenet_layers_output.append({
                "name":
                "enc.layer{}.mlp_residul".format(i),
                "value":
                x.clone()
            })
            np.testing.assert_allclose(x.numpy(),
                                       layer(prev_x, masks, pos_emb,
                                             masks)[0].numpy(),
                                       rtol=1e-7,
                                       atol=1e-10)
        wenet_encoder_out = wenet_model.encoder.after_norm(x)

        # 6. Forward wenet.decoder
        wenet_tokens, _ = add_whisper_tokens(
            configs['tokenizer_conf']['special_tokens'],
            torch.tensor([dummy_tokens], dtype=torch.long),
            ignore_id=-1,
            tasks=[task],
            no_timestamp=True,
            langs=[language],
            use_prev=False)
        L = wenet_tokens.size(1)
        tgt_mask = ~make_pad_mask(torch.tensor([L], dtype=torch.long),
                                  L).unsqueeze(1)  # (B=1, 1, L)
        m = subsequent_mask(tgt_mask.size(-1),
                            device=tgt_mask.device).unsqueeze(0)  # (B=1, L, L)
        tgt_mask = tgt_mask & m  # (B=1, L, L)
        wenet_decoder_embed_posed, wenet_decoder_pos = \
            wenet_model.decoder.embed(wenet_tokens)
        wenet_decoder_embed = wenet_decoder_embed_posed - wenet_decoder_pos
        x = wenet_decoder_embed_posed.clone()
        for i, layer in enumerate(wenet_model.decoder.decoders):
            prev_x = x.clone()
            assert layer.normalize_before
            attn_ln_x = layer.norm1(x)
            wenet_layers_output.append({
                "name": "dec.layer{}.attn_ln".format(i),
                "value": attn_ln_x.clone()
            })
            attn_x = layer.self_attn(attn_ln_x, attn_ln_x, attn_ln_x,
                                     tgt_mask)[0]
            wenet_layers_output.append({
                "name": "dec.layer{}.attn".format(i),
                "value": attn_x.clone()
            })
            x = x + attn_x
            wenet_layers_output.append({
                "name":
                "dec.layer{}.attn_residul".format(i),
                "value":
                x.clone()
            })
            assert layer.src_attn is not None
            assert layer.normalize_before
            cross_attn_ln_x = layer.norm2(x)
            wenet_layers_output.append({
                "name":
                "dec.layer{}.cross_attn_ln".format(i),
                "value":
                cross_attn_ln_x.clone()
            })
            cross_attn_x = layer.src_attn(cross_attn_ln_x, wenet_encoder_out,
                                          wenet_encoder_out, masks)[0]
            wenet_layers_output.append({
                "name":
                "dec.layer{}.cross_attn".format(i),
                "value":
                cross_attn_x.clone()
            })
            x = x + cross_attn_x
            wenet_layers_output.append({
                "name": f"dec.layer{i}.cross_attn_residul",
                "value": x.clone()
            })
            assert layer.normalize_before
            mlp_ln_x = layer.norm3(x)
            wenet_layers_output.append({
                "name": "dec.layer{}.mlp_ln".format(i),
                "value": mlp_ln_x.clone()
            })
            mlp_x = layer.feed_forward(mlp_ln_x)
            wenet_layers_output.append({
                "name": "dec.layer{}.mlp".format(i),
                "value": mlp_x.clone()
            })
            x = x + mlp_x
            wenet_layers_output.append({
                "name":
                "dec.layer{}.mlp_residul".format(i),
                "value":
                x.clone()
            })
            np.testing.assert_allclose(x.numpy(),
                                       layer(prev_x, tgt_mask,
                                             wenet_encoder_out,
                                             masks)[0].numpy(),
                                       rtol=1e-7,
                                       atol=1e-10)
        assert wenet_model.decoder.normalize_before
        x = wenet_model.decoder.after_norm(x)
        assert wenet_model.decoder.use_output_layer
        x = wenet_model.decoder.output_layer(x)
        wenet_logits = x

    np.testing.assert_allclose(whisper_mel.numpy(),
                               wenet_mel.transpose(1, 2).numpy(),
                               rtol=1e-7,
                               atol=1e-10)
    np.testing.assert_allclose(
        whisper_model.encoder.positional_embedding.numpy(),
        pos_emb.squeeze(0).numpy(),
        rtol=1e-7,
        atol=1e-10)
    np.testing.assert_allclose(whisper_subed.numpy(),
                               wenet_subed.numpy(),
                               rtol=1e-7,
                               atol=3e-7)
    np.testing.assert_allclose(whisper_embed.numpy(),
                               wenet_embed.numpy(),
                               rtol=1e-7,
                               atol=1e-10)
    for i, (whisper_layer, wenet_layer) in enumerate(
            zip(whisper_layers_ouput, wenet_layers_output)):
        assert whisper_layer["name"] == wenet_layer["name"]
        print("check layer {}".format(whisper_layer["name"]))
        np.testing.assert_allclose(whisper_layer["value"].numpy(),
                                   wenet_layer["value"].numpy(),
                                   rtol=1e-7,
                                   atol=8e-3)
    np.testing.assert_allclose(whisper_encoder_out.numpy(),
                               wenet_encoder_out.numpy(),
                               rtol=1e-7,
                               atol=6e-03)
    np.testing.assert_allclose(whisper_tokens.numpy(),
                               wenet_tokens.numpy(),
                               rtol=1e-7,
                               atol=1e-10)
    np.testing.assert_allclose(whisper_logits.numpy(),
                               wenet_logits.numpy(),
                               rtol=1e-7,
                               atol=6e-02)
    np.testing.assert_allclose(F.softmax(whisper_logits).numpy(),
                               F.softmax(wenet_logits).numpy(),
                               rtol=1e-7,
                               atol=1e-10)
