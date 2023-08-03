from wenet.dataset.processor import __tokenize_by_bpe_model
from typing import Dict, List


def tokenize(context_list_path, symbol_table, bpe_model=None):
    """ Read biasing list from the biasing list address, tokenize and convert it
        into token id
    """
    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    with open(context_list_path, "r") as fin:
        context_txts = fin.readlines()

    context_list = []
    for context_txt in context_txts:
        context_txt = context_txt.strip()

        labels = []
        tokens = []
        if bpe_model is not None:
            tokens = __tokenize_by_bpe_model(sp, context_txt)
        else:
            for ch in context_txt:
                if ch == ' ':
                    ch = "▁"
                tokens.append(ch)
        for ch in tokens:
            if ch in symbol_table:
                labels.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                labels.append(symbol_table['<unk>'])
        context_list.append(labels)
    return context_list


class ContextGraph:
    """ Context decoding graph, constructing graph using dict instead of WFST
        Args:
            context_list_path(str): context list path
            bpe_model(str): model for english bpe part
            context_score(float): context score for each token
    """
    def __init__(self,
                 context_list_path: str,
                 symbol_table: Dict[str, int],
                 bpe_model: str = None,
                 context_score: float = 6):
        self.context_score = context_score
        self.context_list = tokenize(context_list_path, symbol_table,
                                     bpe_model)
        self.graph = {0: {}}
        self.graph_size = 0
        self.state2token = {}
        self.back_score = {0: 0.0}
        self.build_graph(self.context_list)

    def build_graph(self, context_list: List[List[int]]):
        """ Constructing the context decoding graph, add arcs with negative
            scores returning to the starting state for each non-terminal tokens
            of hotwords, and add arcs with scores of 0 returning to the starting
            state for terminal tokens.
        """
        self.graph = {0: {}}
        self.graph_size = 0
        self.state2token = {}
        self.back_score = {0: 0.0}
        for context_token in context_list:
            now_state = 0
            for i in range(len(context_token)):
                if context_token[i] in self.graph[now_state]:
                    now_state = self.graph[now_state][context_token[i]]
                    if i == len(context_token) - 1:
                        self.back_score[now_state] = 0
                else:
                    self.graph_size += 1
                    self.graph[self.graph_size] = {}
                    self.graph[now_state][context_token[i]] = self.graph_size
                    now_state = self.graph_size
                    if i != len(context_token) - 1:
                        self.back_score[now_state] = -(i +
                                                       1) * self.context_score
                    else:
                        self.back_score[now_state] = 0
                    self.state2token[now_state] = context_token[i]

    def find_next_state(self, now_state: int, token: int):
        """ Search for an arc with the input being a token from the current state,
            returning the score on the arc and the state it points to. If there is
            no match, return to the starting state and perform an additional search
            from the starting state to avoid token consumption due to mismatches.
        """
        if token in self.graph[now_state]:
            return self.graph[now_state][token], self.context_score
        back_score = self.back_score[now_state]
        now_state = 0
        if token in self.graph[now_state]:
            return self.graph[now_state][
                token], back_score + self.context_score
        return 0, back_score
