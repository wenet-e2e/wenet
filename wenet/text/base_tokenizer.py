from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List, Union, Any

T = Union[str, bytes]


class BaseTokenizer(ABC):

    def tokenize(self, line: str) -> Dict[str, Any]:
        tokens = self.text2tokens(line)
        ids = self.tokens2ids(tokens)
        return {
            "tokens": {
                "ctc": tokens,
                "decoder": tokens
            },
            "label": {
                "ctc": ids,
                "decoder": ids
            }
        }

    def detokenize(self, ids: List[int]) -> Dict[str, Any]:
        tokens = self.ids2tokens(ids)
        text = self.tokens2text(tokens)
        return {
            "text": {
                "ctc": text,
                "decoder": text
            },
            "tokens": {
                "ctc": tokens,
                "decoder": tokens
            }
        }

    @abstractmethod
    def text2tokens(self, line: str) -> List[T]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2text(self, tokens: List[T]) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2ids(self, tokens: List[T]) -> List[int]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def ids2tokens(self, ids: List[int]) -> List[T]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError("abstract method")

    @abstractproperty
    def symbol_table(self) -> Dict[T, int]:
        raise NotImplementedError("abstract method")
