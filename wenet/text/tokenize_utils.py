# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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

import re


def tokenize_by_bpe_model(sp, txt):
    return _tokenize_by_seg_dic_or_bpe_model(txt, sp=sp, upper=True)


def tokenize_by_seg_dict(seg_dict, txt):
    return _tokenize_by_seg_dic_or_bpe_model(txt,
                                             seg_dict=seg_dict,
                                             upper=False)


def _tokenize_by_seg_dic_or_bpe_model(
    txt,
    sp=None,
    seg_dict=None,
    upper=True,
):
    if sp is None:
        assert seg_dict is not None
    if seg_dict is None:
        assert sp is not None
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper() if upper else txt)
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            if sp is not None:
                for p in sp.encode_as_pieces(ch_or_w):
                    tokens.append(p)
            else:
                for en_token in ch_or_w.split():
                    en_token = en_token.strip()
                    if en_token in seg_dict:
                        tokens.extend(seg_dict[en_token].split(' '))
                    else:
                        tokens.append(en_token)

    return tokens
