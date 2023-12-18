# Copyright (c) 2023 Wenet Community. (authors: Xingchen Song)
#
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

from typing import Dict, List, Any

from wenet.text.base_tokenizer import BaseTokenizer


class HybridTokenizer():

    def __init__(self, tokenizers: Dict[str, BaseTokenizer]) -> None:
        self.tokenizers = tokenizers

    def tokenize(self, line: str) -> Dict[str, Any]:
        result = {"tokens": {}, "label": {}}
        for module, tokenizer in self.tokenizers.items():
            tokens = tokenizer.text2tokens(line)
            ids = tokenizer.tokens2ids(tokens)
            result["tokens"][module] = tokens
            result["label"][module] = ids
        return result

    def detokenize(self, ids: List[int]) -> Dict[str, Any]:
        result = {"tokens": {}, "text": {}}
        for module, tokenizer in self.tokenizers.items():
            tokens = tokenizer.ids2tokens(ids)
            text = tokenizer.tokens2text(tokens)
            result["text"][module] = text
            result["tokens"][module] = tokens
        return result

    def vocab_size(self) -> Dict[str, int]:
        result = {}
        for module in self.tokenizers.keys():
            result[module] = len(self.tokenizers[module].char_dict)
        return result

    @property
    def symbol_table(self) -> Dict[str, Dict[str, int]]:
        result = {}
        for module in self.tokenizers.keys():
            result[module] = self.tokenizers[module].symbol_table
        return result
