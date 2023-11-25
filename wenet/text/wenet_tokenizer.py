import re

from collections.abc import Iterable
from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.text.base_tokenizer import BaseTokenizer
from wenet.text.tokenize_utils import tokenize_by_bpe_model


class WenetTokenizer(BaseTokenizer):
    """Wrapper for original wenet tokenize implementation
    """

    def __init__(
        self,
        symbol_table: Union[str, PathLike, Dict],
        bpe_model: Optional[Union[str, PathLike]] = None,
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
    ) -> None:
        self.non_lang_syms_pattern = None
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(
                r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        if not isinstance(symbol_table, Dict):
            self.symbol_table = read_symbol_table(symbol_table)
        else:
            # symbol_table = {"我": 1, "是": 2, "{NOISE}": 3}
            self.symbol_table = symbol_table
        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        self.bpe_model = None
        if bpe_model is not None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(bpe_model)
        self.char_dict = {v: k for k, v in self.symbol_table.items()}
        self.split_with_space = split_with_space
        self.connect_symbol = connect_symbol

    def text2tokens(self, line: str) -> List[str]:
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.bpe_model is not None:
                    tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
                else:
                    if self.split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.connect_symbol.join(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for ch in tokens:
            if ch in self.symbol_table:
                ids.append(self.symbol_table[ch])
            elif '<unk>' in self.symbol_table:
                ids.append(self.symbol_table['<unk>'])
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        content = [self.char_dict[w] for w in ids]
        return content

    def vocab_size(self) -> int:
        return len(self.char_dict)
