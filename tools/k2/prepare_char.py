#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                  Wei Kang)
# Copyright    2022  Ximalaya Speech Team (author: Xiang Lyu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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


"""

This script generates the following files in the directory sys.argv[3]:

    - lexicon.txt
    - lexicon_disambig.txt
    - L.pt
    - L_disambig.pt
    - tokens.txt
    - words.txt
"""

import sys
from pathlib import Path
from typing import Dict, List

import k2
import torch
from prepare_lang import (
    Lexicon,
    add_disambig_symbols,
    add_self_loops,
    write_lexicon,
    write_mapping,
)


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # The blank symbol <blk> is defined in local/train_bpe_model.py
    assert token2id["<blank>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [
            token2id[i] if i in token2id else token2id["<unk>"] for i in pieces
        ]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def contain_oov(token_sym_table: Dict[str, int], tokens: List[str]) -> bool:
    """Check if all the given tokens are in token symbol table.

    Args:
      token_sym_table:
        Token symbol table that contains all the valid tokens.
      tokens:
        A list of tokens.
    Returns:
      Return True if there is any token not in the token_sym_table,
      otherwise False.
    """
    for tok in tokens:
        if tok not in token_sym_table:
            return True
    return False


def generate_lexicon(
    token_sym_table: Dict[str, int], words: List[str]
) -> Lexicon:
    """Generate a lexicon from a word list and token_sym_table.

    Args:
      token_sym_table:
        Token symbol table that mapping token to token ids.
      words:
        A list of strings representing words.
    Returns:
      Return a dict whose keys are words and values are the corresponding
          tokens.
    """
    lexicon = []
    for word in words:
        chars = list(word.strip(" \t"))
        if contain_oov(token_sym_table, chars):
            continue
        lexicon.append((word, chars))

    # The OOV word is <UNK>
    lexicon.append(("<UNK>", ["<unk>"]))
    return lexicon


def generate_tokens(text_file: str) -> Dict[str, int]:
    """Generate tokens from the given text file.

    Args:
      text_file:
        A file that contains text lines to generate tokens.
    Returns:
      Return a dict whose keys are tokens and values are token ids ranged
      from 0 to len(keys) - 1.
    """
    token2id: Dict[str, int] = dict()
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            char, index = line.replace('\n', '').split()
            assert char not in token2id
            token2id[char] = int(index)
    assert token2id['<blank>'] == 0
    return token2id


def generate_words(text_file: str) -> Dict[str, int]:
    """Generate words from the given text file.

    Args:
      text_file:
        A file that contains text lines to generate words.
    Returns:
      Return a dict whose keys are words and values are words ids ranged
      from 0 to len(keys) - 1.
    """
    words = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            word = line.replace('\n', '')
            assert word not in words
            words.append(word)
    words.sort()

    # We put '<eps>' '<UNK>' at begining of word2id
    # '#0', '<s>', '</s>' at end of word2id
    words = [word for word in words
             if word not in ['<eps>', '<UNK>', '#0', '<s>', '</s>']]
    words.insert(0, '<eps>')
    words.insert(1, '<UNK>')
    words.append('#0')
    words.append('<s>')
    words.append('</s>')
    word2id = {j: i for i, j in enumerate(words)}
    return word2id


def main():
    token2id = generate_tokens(sys.argv[1])
    word2id = generate_words(sys.argv[2])
    tgt_dir = Path(sys.argv[3])

    words = [word for word in word2id.keys()
             if word not in
             ["<eps>", "!SIL", "<SPOKEN_NOISE>", "<UNK>", "#0", "<s>", "</s>"]]
    lexicon = generate_lexicon(token2id, words)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)
    next_token_id = max(token2id.values()) + 1
    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in token2id
        token2id[disambig] = next_token_id
        next_token_id += 1

    write_mapping(tgt_dir / "tokens.txt", token2id)
    write_mapping(tgt_dir / "words.txt", word2id)
    write_lexicon(tgt_dir / "lexicon.txt", lexicon)
    write_lexicon(tgt_dir / "lexicon_disambig.txt", lexicon_disambig)

    L = lexicon_to_fst_no_sil(
        lexicon,
        token2id=token2id,
        word2id=word2id,
    )
    L_disambig = lexicon_to_fst_no_sil(
        lexicon_disambig,
        token2id=token2id,
        word2id=word2id,
        need_self_loops=True,
    )
    torch.save(L.as_dict(), tgt_dir / "L.pt")
    torch.save(L_disambig.as_dict(), tgt_dir / "L_disambig.pt")


if __name__ == "__main__":
    main()
