from os import PathLike

from typing import Dict, List, Optional, Union

from wenet.text.char_tokenizer import CharTokenizer
from wenet.text.tokenize_utils import tokenize_by_bpe_model


class BpeTokenizer(CharTokenizer):

    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
        upper: bool = True,
    ) -> None:
        super().__init__(symbol_table, non_lang_syms, split_with_space,
                         connect_symbol, unk)
        self._model = bpe_model
        # NOTE(Mddct): multiprocessing.Process() issues
        #              don't build sp here
        self.bpe_model = None
        # NOTE(Mddct): we can handle proto, see:
        # https://github.com/google/sentencepiece/issues/121#issuecomment-400362011
        self.bpe_spm = None
        self.upper = upper
        self.extra_tokens = {}

    def _build_sp(self):
        import sentencepiece as spm
        if self.bpe_model is None:
            self.bpe_model = spm.SentencePieceProcessor()
            self.bpe_model.Load(self._model)
            if len(self.extra_tokens) > 0:
                from transformers.utils import (sentencepiece_model_pb2_new as
                                                sentencepiece_model_pb2)
                self.bpe_spm = sentencepiece_model_pb2.ModelProto()
                self.bpe_spm.ParseFromString(
                    self.bpe_model.serialized_model_proto())
                for token_id in sorted(self.extra_tokens.items(),
                                       key=lambda x: x[1]):
                    new_p = sentencepiece_model_pb2.ModelProto().SentencePiece(
                    )
                    new_p.piece = token_id[0]
                    new_p.score = 0
                    self.bpe_spm.pieces.append(new_p)

                self.bpe_model = spm.SentencePieceProcessor(
                    model_proto=self.bpe_spm.SerializeToString())

    def text2tokens(self, line: str) -> List[str]:
        self._build_sp()
        line = line.strip()
        line = line.upper() if self.upper else line
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line)
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part == '':
                continue
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
            token = token.upper() if self.upper else token
            if token not in self.symbol_table:
                self.symbol_table[token] = len(self.symbol_table)
                added_tokens += 1
                self.char_dict[len(self.char_dict)] = token
                self.extra_tokens[token] = self.symbol_table[token]
        return added_tokens
