from os import PathLike
from typing import Dict, List, Optional, Union
from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_bpe_model
from wenet.text.tries import Trie


class BpeTokenizer(CharTokenizer):

    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        super().__init__(symbol_table, non_lang_syms, split_with_space,
                         connect_symbol, unk)
        self._model = bpe_model
        # NOTE(Mddct): multiprocessing.Process() issues
        #              don't build sp here
        self.bpe_model = None
        # NOTE(Mddct): we can handle proto, see:
        # https://github.com/google/sentencepiece/issues/121#issuecomment-400362011
        self.extra_tokens = Trie()

    def _build_sp(self):
        if self.bpe_model is None:
            import sentencepiece as spm
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.load(self._model)

    def text2tokens(self, line: str) -> List[str]:
        self._build_sp()
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        extra_part = []
        for part in parts:
            if part in self.non_lang_syms:
                extra_part.append(part)
            else:
                extra_part.extend(self.extra_tokens.split(part))
        parts = extra_part
        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(tokenize_by_bpe_model(self.bpe_model, part))
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        self._build_sp()
        text = super().tokens2text(tokens)
        return text.replace("▁", ' ').strip()

    def add_tokens(self, tokens: List[str]) -> int:
        added_tokens = 0
        for token in tokens:
            if token not in self.symbol_table:
                self.extra_tokens.add(token)
                self.symbol_table[token] = len(self.symbol_table)
                added_tokens += 1
                self.char_dict[len(self.char_dict)] = token
        return added_tokens
