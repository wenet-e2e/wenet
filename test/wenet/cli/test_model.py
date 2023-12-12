#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-12] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import os
import requests
import pytest

from wenet.cli.hub import download
from wenet.cli.model import Model


@pytest.mark.parametrize("model", [
    "aishell_u2pp_conformer_libtorch.tar.gz",
    "aishell2_u2pp_conformer_libtorch.tar.gz",
    "gigaspeech_u2pp_conformer_libtorch.tar.gz",
    "librispeech_u2pp_conformer_libtorch.tar.gz",
    "multi_cn_unified_conformer_libtorch.tar.gz",
    "wenetspeech_u2pp_conformer_libtorch.tar.gz"
])
def test_model(model):
    dest = model.split('.')[0]  # aishell_u2pp_conformer_libtorch
    dataset = model.split('_')[0]  # aishell
    if not os.path.exists(dest):
        os.makedirs(dest)
    response = requests.get(
        "https://modelscope.cn/api/v1/datasets/wenet/wenet_pretrained_models/oss/tree"  # noqa
    )
    model_info = next(data for data in response.json()["Data"]
                      if data["Key"] == model)
    model_url = model_info['Url']
    download(model_url, dest=dest, only_child=True)
    model = Model(dest, gpu=-1, beam=5, resample_rate=16000)
    if dataset in ['gigaspeech', 'librispeech']:
        audio_file = "test/resources/librispeech-1995-1837-0001.wav"
        text = "▁IT▁WAS▁THE▁FIRST▁GREAT▁SORROW▁OF▁HIS▁LIFE▁IT▁WAS▁NOT▁SO▁MUCH" + \
            "▁THE▁LOSS▁OF▁THE▁COTTON▁ITSELF▁BUT▁THE▁FANTASY▁THE▁HOPES▁THE▁DREAMS▁BUILT▁AROUND▁IT"
    else:
        audio_file = "test/resources/aishell-BAC009S0724W0121.wav"
        text = "广州市房地产中介协会分析"
    result = model.transcribe(audio_file)
    print(result)
    assert result['text'] == text
