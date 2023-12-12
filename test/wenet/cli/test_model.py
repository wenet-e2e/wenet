#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-12-12] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import pytest

from wenet.cli.hub import download
from wenet.cli.model import Model


@pytest.mark.parametrize("model_link", [
    "https://wenet.org.cn/downloads?models=wenet&version=aishell_u2pp_conformer_libtorch.tar.gz",
    "https://wenet.org.cn/downloads?models=wenet&version=aishell2_u2pp_conformer_libtorch.tar.gz",
    "https://wenet.org.cn/downloads?models=wenet&version=gigaspeech_u2pp_conformer_libtorch.tar.gz",
    "https://wenet.org.cn/downloads?models=wenet&version=librispeech_u2pp_conformer_libtorch.tar.gz",
    "https://wenet.org.cn/downloads?models=wenet&version=multi_cn_unified_conformer_libtorch.tar.gz",
    "https://wenet.org.cn/downloads?models=wenet&version=wenetspeech_u2pp_conformer_libtorch.tar.gz"
])
def test_model(model_link):
    dest = model_link.split('=')[-1].split('.')[0]  # aishell_u2pp_conformer_libtorch
    dataset = model_link.split('_')[-4].split('=')[-1]  # aishell
    download(model_link, dest=dest)
    model = Model(model_link,
                  gpu=-1,
                  beam=5,
                  resample_rate=16000)
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
