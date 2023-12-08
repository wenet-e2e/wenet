# Copyright (c) 2023 Wenet Community. (authors: Dinghao Zhou)
#                                     (authors: Xingchen Song)
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

import logging

from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.bpe_tokenizer import BpeTokenizer
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.paraformer_tokenizer import ParaformerTokenizer
from wenet.text.whisper_tokenizer import WhisperTokenizer


def init_tokenizer(configs) -> BaseTokenizer:
    # TODO(Mddct):
    # 1 paraformer tokenizer

    if configs["tokenizer"] == "whisper":
        tokenizer = WhisperTokenizer(
            multilingual=configs['tokenizer_conf']['is_multilingual'],
            num_languages=configs['tokenizer_conf']['num_languages'])
    elif configs["tokenizer"] == "char":
        tokenizer = CharTokenizer(
            configs['tokenizer_conf']['symbol_table_path'],
            configs['tokenizer_conf']['non_lang_syms_path'],
            split_with_space=configs['tokenizer_conf'].get(
                'split_with_space', False))
    elif configs["tokenizer"] == "bpe":
        tokenizer = BpeTokenizer(
            configs['tokenizer_conf']['bpe_path'],
            configs['tokenizer_conf']['symbol_table_path'],
            configs['tokenizer_conf']['non_lang_syms_path'],
            split_with_space=configs['tokenizer_conf'].get(
                'split_with_space', False))
    elif configs["tokenizer"] == 'paraformer':
        assert 'tokenizer' in configs
        assert 'tokenizer_conf' in configs
        assert symbol_table == configs['tokenizer_conf']['symbol_table_path']
        return ParaformerTokenizer(
            symbol_table=configs['tokenizer_conf']['symbol_table_path'],
            seg_dict=configs['tokenizer_conf']['seg_dict_path'])
    else:
        raise NotImplementedError
    logging.info("use {} tokenizer".format(configs["tokenizer"]))

    return tokenizer
