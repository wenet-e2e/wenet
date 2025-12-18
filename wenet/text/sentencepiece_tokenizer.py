from os import PathLike
from typing import Dict, List, Union

from wenet.text.base_tokenizer import BaseTokenizer, T


class SentencepieceTokenizer(BaseTokenizer):
    """ Sentencepiece Tokenizer
    """

    def __init__(
        self,
        model_path: Union[PathLike, str],
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_path = model_path
        self.model = None
        self._vocab_size = None
        self._symbol_table = None

    def _build_sp(self):
        if self.model is None:
            import sentencepiece as spm
            self.model = spm.SentencePieceProcessor()
            self.model.load(self.model_path)
            self._symbol_table = {
                self.model.id_to_piece(_id): _id
                for _id in range(self.model.get_piece_size())
            }
            self.vocab_size = len(self._symbol_table)

    def text2tokens(self, line: str) -> List[T]:
        self._build_sp()
        return self.model.encode_as_pieces(line)

    def tokens2ids(self, tokens: List[T]) -> List[int]:
        self._build_sp()
        return self.model.piece_to_id(tokens)

    def ids2tokens(self, ids: List[int]) -> List[T]:
        self._build_sp()
        return self.model.id_to_piece(ids)

    def tokens2text(self, tokens: List[T]) -> str:
        self._build_sp()
        return self.model.decode(tokens)

    @property
    def symbol_table(self) -> Dict[T, int]:
        self._build_sp()
        return self._symbol_table

    def vocab_size(self) -> int:
        self._build_sp()
        return self.vocab_size
