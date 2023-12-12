from os import PathLike
import re
from typing import Dict, List, Optional, Union
from wenet.text.char_tokenizer import CharTokenizer


def read_seg_dict(path):
    seg_table = {}
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            assert len(arr) == 2
            seg_table[arr[0]] = arr[1]
    return seg_table


class ParaformerTokenizer(CharTokenizer):

    def __init__(self,
                 symbol_table: Union[str, PathLike, Dict],
                 seg_dict: Optional[Union[str, PathLike, Dict]] = None,
                 split_with_space: bool = False,
                 connect_symbol: str = '',
                 unk='<unk>') -> None:
        super().__init__(symbol_table, None, split_with_space, connect_symbol,
                         unk)
        self.seg_dict = seg_dict
        if seg_dict is not None and not isinstance(seg_dict, Dict):
            self.seg_dict = read_seg_dict(seg_dict)

    def text2tokens(self, line: str) -> List[str]:
        assert self.seg_dict is not None

        # TODO(Mddct): duplicated here, refine later
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line)
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(self.tokenize_by_seg_dict(part))
        return tokens

    def tokenize_by_seg_dict(self, txt):
        tokens = []
        # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        pattern = re.compile(r'([\u4e00-\u9fff])')
        # Example:
        #   txt   = "你好 ITS'S OKAY 的"
        #   chars = ["你", "好", " ITS'S OKAY ", "的"]
        chars = pattern.split(txt)
        mix_chars = [w for w in chars if len(w.strip()) > 0]
        for ch_or_w in mix_chars:
            # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
            if pattern.fullmatch(ch_or_w) is not None:
                tokens.append(ch_or_w)
            # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
            # encode ch_or_w using bpe_model.
            else:
                for en_token in ch_or_w.split():
                    en_token = en_token.strip()
                    if en_token in self.seg_dict:
                        tokens.extend(self.seg_dict[en_token].split(' '))
                    else:
                        tokens.append(en_token)

        return tokens
