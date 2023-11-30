from os import PathLike
from typing import Union
from wenet.text.base_tokenizer import BaseTokenizer


class HuggingFaceTokenizer(BaseTokenizer):

    def __init__(self, model: Union[str, PathLike]) -> None:
        # NOTE(Mddct): don't build here, pickle issues
        self.model = model
        self.tokenizer = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['tokenizer']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        recovery = {'tokenizer': None}
        self.__dict__.update(recovery)

    def _build_hugging_face(self):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.t2i = {}
            self.i2t = {}
            for i in range(self.tokenizer.encoding.n_vocab):
                unit = str(
                    self.tokenizer.encoding.decode_single_token_bytes(i))
                if len(unit) == 0:
                    unit = str(i)
                unit = unit.replace(" ", "<space>")
                # unit = bytes(unit, 'utf-8')
                self.t2i[unit] = i
                self.i2t[i] = unit
            assert len(self.t2i) == len(self.i2t)

    def text2tokens(self, line: str) -> List[str]:
        self._build_hugging_face()
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: List[str]) -> str:
        self._build_hugging_face()
        ids = self.tokens2ids(tokens)
        return self.tokenizer.decode(ids)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        self._build_hugging_face()
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def ids2tokens(self, ids: List[int]) -> List[str]:
        self._build_hugging_face()
        return self.tokenizer.convert_ids_to_tokens(ids)

    def vocab_size(self) -> int:
        self._build_hugging_face()
        # TODO: we need special tokenize size in future
        return len(self.tokenizer)

    @property
    def symbol_table(self) -> Dict[str, int]:
        self._build_tiktoken()
        return self.t2i
