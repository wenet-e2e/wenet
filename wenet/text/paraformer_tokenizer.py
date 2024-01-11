from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.paraformer.search import paraformer_beautify_result
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_seg_dict


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
                tokens.extend(tokenize_by_seg_dict(self.seg_dict, part))
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        return paraformer_beautify_result(tokens)
