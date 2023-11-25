from collections.abc import Iterable
from os import PathLike
from typing import List, Optional, Tuple, Union
from wenet.text.base_tokenizer import BaseTokenizer
from whisper.tokenizer import get_tokenizer

from wenet.utils.file_utils import read_non_lang_symbols


class WhisperTokenizer(BaseTokenizer):

    def __init__(
        self,
        multilingual: bool,
        num_languages: int = 99,
        language: Optional[str] = None,
        task: Optional[str] = None,
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        *args,
        **kwargs,
    ) -> None:
        self.tokenizer = get_tokenizer(multilingual=multilingual,
                                       num_languages=num_languages,
                                       language=language,
                                       task=task)
        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        # TODO(Mddct): add special tokens, like non_lang_syms
        del self.non_lang_syms
        self.t2i = {}
        self.i2t = {}
        for i in range(self.tokenizer.encoding.n_vocab):
            unit = str(self.tokenizer.encoding.decode_single_token_bytes(i))
            if len(unit) == 0:
                unit = str(i)
            unit = unit.replace(" ", "<space>")
            # unit = bytes(unit, 'utf-8')
            self.t2i[unit] = i
            self.i2t[i] = unit
        print(len(self.t2i), len(self.i2t))
        assert len(self.t2i) == len(self.i2t)

    def tokenize(self, line: str) -> Tuple[List[str], List[int]]:
        ids = self.tokenizer.encoding.encode(line)
        text = [self.i2t[d] for d in ids]
        return text, ids

    def detokenize(self, ids: List[int]) -> Tuple[str, List[str]]:
        tokens = [self.i2t[d] for d in ids]
        text = self.tokenizer.encoding.decode(ids)
        return text, tokens

    def text2tokens(self, line: str) -> List[str]:
        return self.tokenize(line)[0]

    def tokens2text(self, tokens: Iterable[str]) -> str:
        ids = [self.t2i[t] for t in tokens]
        return self.detokenize(ids)[0]

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = [self.t2i[t] for t in tokens]
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        return [self.tokenizer.encoding.decode([id]) for id in ids]

    def vocab_size(self) -> int:
        return len(self.t2i)
