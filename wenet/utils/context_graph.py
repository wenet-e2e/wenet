import torch
from torch.nn.utils.rnn import pad_sequence

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

def tbbm(sp, context_txt):
    return __tokenize_by_bpe_model(sp, context_txt)

class ContextGraph:
    """ Context decoding graph, constructing graph using dict instead of WFST
        Args:
            context_list_path(str): context list path
            bpe_model(str): model for english bpe part
            context_graph_score(float): context score for each token
    """
    def __init__(self,
                 context_list_path: str,
                 symbol_table: Dict[str, int],
                 bpe_model: str = None,
                 context_graph_score: float = 2.0):
        self.context_graph_score = context_graph_score
        self.context_list = tokenize(context_list_path, symbol_table,
                                     bpe_model)
        self.graph = {0: {}}
        self.graph_size = 0
        self.state2token = {}
        self.back_score = {0: 0.0}
        self.build_graph(self.context_list)
        self.graph_biasing = False
        self.deep_biasing = False
        self.deep_biasing_score = 1.0
        self.context_filtering = True

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
                        self.back_score[now_state] = \
                            -(i + 1) * self.context_graph_score
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
            return self.graph[now_state][token], self.context_graph_score
        back_score = self.back_score[now_state]
        now_state = 0
        if token in self.graph[now_state]:
            return self.graph[now_state][token], \
                back_score + self.context_graph_score
        return 0, back_score

    def get_context_list_tensor(self, context_list: List[List[int]]):
        context_list_tensor = [torch.tensor([0], dtype=torch.int32)]
        for context_token in context_list:
            context_list_tensor.append(torch.tensor(context_token, dtype=torch.int32))
        context_list_lengths = torch.tensor([x.size(0) for x in context_list_tensor],
                                            dtype=torch.int32)
        context_list_tensor = pad_sequence(context_list_tensor,
                                           batch_first=True,
                                           padding_value=-1)
        return context_list_tensor, context_list_lengths

    def two_stage_filtering(self,
                            context_list: List[List[int]],
                            ctc_posterior: torch.Tensor,
                            filter_threshold: float = -4,
                            filter_window_size: int = 64):
        if len(context_list) == 0:
            return context_list

        SOC_score = {}
        for t in range(1, ctc_posterior.shape[0]):
            if t % (filter_window_size // 2) != 0 and t != ctc_posterior.shape[0] - 1:
                continue
            # calculate PSC
            PSC_score = {}
            max_posterior, _ = torch.max(ctc_posterior[max(0,
                                         t - filter_window_size):t, :],
                                         dim=0, keepdim=False)
            max_posterior = max_posterior.tolist()
            for i in range(len(context_list)):
                score = sum(max_posterior[j] for j in context_list[i]) \
                    / len(context_list[i])
                PSC_score[i] = max(SOC_score.get(i, -float('inf')), score)
            PSC_filtered_index = []
            for i in PSC_score:
                if PSC_score[i] > filter_threshold:
                    PSC_filtered_index.append(i)
            if len(PSC_filtered_index) == 0:
                continue
            filtered_context_list = []
            for i in PSC_filtered_index:
                filtered_context_list.append(context_list[i])

            # calculate SOC
            win_posterior = ctc_posterior[max(0, t - filter_window_size):t, :]
            win_posterior = win_posterior.unsqueeze(0) \
                .expand(len(filtered_context_list), -1, -1)
            select_win_posterior = []
            for i in range(len(filtered_context_list)):
                select_win_posterior.append(torch.index_select(
                    win_posterior[0], 1,
                    torch.tensor(filtered_context_list[i],
                                 device=ctc_posterior.device)).transpose(0, 1))
            select_win_posterior = \
                pad_sequence(select_win_posterior,
                             batch_first=True).transpose(1, 2).contiguous()
            dp = torch.full((select_win_posterior.shape[0],
                             select_win_posterior.shape[2]),
                            -10000.0, dtype=torch.float32,
                            device=select_win_posterior.device)
            dp[:, 0] = select_win_posterior[:, 0, 0]
            for win_t in range(1, select_win_posterior.shape[1]):
                temp = dp[:, :-1] + select_win_posterior[:, win_t, 1:]
                idx = torch.where(temp > dp[:, 1:])
                idx_ = (idx[0], idx[1] + 1)
                dp[idx_] = temp[idx]
                dp[:, 0] = \
                    torch.where(select_win_posterior[:, win_t, 0] > dp[:, 0],
                                select_win_posterior[:, win_t, 0], dp[:, 0])
            for i in range(len(filtered_context_list)):
                SOC_score[PSC_filtered_index[i]] = \
                    max(SOC_score.get(PSC_filtered_index[i], -float('inf')),
                        dp[i][len(filtered_context_list[i]) - 1]
                        / len(filtered_context_list[i]))
        filtered_context_list = []
        for i in range(len(context_list)):
            if SOC_score.get(i, -float('inf')) > filter_threshold:
                filtered_context_list.append(context_list[i])
        return filtered_context_list
