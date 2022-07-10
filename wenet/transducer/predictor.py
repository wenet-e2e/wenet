from typing import Optional, Tuple

import torch
from torch import nn
from torch.random import initial_seed
from typeguard import check_argument_types
from wenet.utils.common import get_activation


# TODO(Mddct): move to utils
def get_rnn(rnn_type: str) -> nn.Module:
    assert rnn_type in ["rnn", "lstm", "gru"]
    if rnn_type == "rnn":
        return torch.nn.RNN
    elif rnn_type == "lstm":
        return torch.nn.LSTM
    else:
        return torch.nn.GRU


def ApplyPadding(input, padding, pad_value) -> torch.Tensor:
    """
    Args:
        input:   [bs, max_time_step, dim]
        padding: [bs, max_time_step]
    """
    return padding * pad_value + input * (1 - padding)


class RNNPredictor(torch.nn.Module):

    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 output_size: int,
                 embed_dropout: float,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 rnn_type: str = "lstm",
                 dropout: float = 0.1) -> None:
        assert check_argument_types()
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        # disable rnn base out projection
        self.embed = nn.Embedding(voca_size, embed_size)
        self.dropout = nn.Dropout(embed_dropout)
        # NOTE(Mddct): rnn base from torch not support layer norm
        # will add layer norm and prune value in cell and layer
        # ref: https://github.com/Mddct/neural-lm/blob/main/models/gru_cell.py
        self.rnn = get_rnn(rnn_type=rnn_type)(input_size=embed_size,
                                              hidden_size=hidden_size,
                                              num_layers=num_layers,
                                              bias=bias,
                                              batch_first=True,
                                              dropout=dropout)
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        state: Optional[Tuple[torch.Tensor,
                              torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): [batch, max_time).
            padding (torch.Tensor): [batch, max_time]
        Returns:
            output: [batch, max_time, output_size]
        """

        # NOTE(Mddct): we don't use pack input format
        embed = self.embed(input)  # [batch, max_time, emb_size]
        embed = self.dropout(embed)
        if state is None:
            state = self.init_state(batch_size=input.size(0))
            state = (state[0].to(input.device), state[1].to(input.device))
        out, (m, c) = self.rnn(embed, state)
        out = self.projection(out)

        # NOTE(Mddct): Although we don't use staate in transducer
        # training forward, we need make it right for padding value
        # so we create forward_step for infering, forward for training
        _, _ = m, c
        return out

    def init_state(self,
                   batch_size: int,
                   method: str = "zero") -> Tuple[torch.Tensor, torch.Tensor]:
        assert batch_size > 0
        # TODO(Mddct): xavier init method
        _ = method
        return torch.zeros(1 * self.n_layers, batch_size,
                           self.hidden_size), torch.zeros(
                               1 * self.n_layers, batch_size, self.hidden_size)

    def forward_step(
        self, input: torch.Tensor, padding: torch.Tensor,
        state_m: torch.Tensor, state_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            padding (torch.Tensor): [batch_size,1], 1 is padding value
        """
        embed = self.embed(input)  # [batch, 1, emb_size]
        embed = self.dropout(embed)
        out, (m, c) = self.rnn(embed, (state_m, state_c))

        out = self.projection(out)
        m = ApplyPadding(m, padding, state_m)
        c = ApplyPadding(c, padding, state_c)
        return out, m, c


class EmbeddingPredictor(torch.nn.Module):
    """Embedding predictor

    Described in:
    https://arxiv.org/pdf/2109.07513.pdf

    embed-> proj -> layer norm -> swish
    """

    def __init__(self,
                 voca_size: int,
                 embed_size: int,
                 embed_dropout: float,
                 n_head: int,
                 history_size: int = 2,
                 activation: str = "swish",
                 layer_norm_epsilon: float = 1e-5) -> None:

        assert check_argument_types()
        super().__init__()
        # multi head
        self.num_heads = n_head
        self.embed_size = embed_size
        self.context_size = history_size + 1
        self.pos_embed = torch.nn.Linear(n_head,
                                         embed_size * self.context_size,
                                         bias=False)
        self.embed = nn.Embedding(voca_size, self.embed_size)
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.ffn = nn.Linear(self.embed_size, self.embed_size)
        self.norm = nn.LayerNorm(self.embed_size, eps=layer_norm_epsilon)
        self.activatoin = get_activation(activation)

    @property
    def embedding_weight(self) -> torch.Tensor:
        # To see how it works:
        return self.embed.weight

    def init_state(self,
                   batch_size: int,
                   method: str = "zero") -> torch.Tensor:
        assert batch_size > 0
        _ = method
        return torch.zeros(batch_size, self.context_size - 1, self.embed_size)

    def forward(self, input: torch.Tensor):
        """ forward for training
        """
        input = self.embed(input)  # [bs, seq_len, embed]
        # zeros = torch.zeros(input.size(0), self.history_size,
        #                     self.embed_size).to(input.device)
        zeros = self.init_state(input.size(0))
        input = torch.cat((zeros, input),
                          dim=1)  # [bs, context_size-1 + seq_len, embed]

        input = input.unfold(1, self.context_size, 1).permute(
            0, 1, 3, 2)  # [bs, seq_len, context_size, embed]
        # multi head pos: [n_head, embed, context_size]
        multi_head_pos = self.pos_embed.weight.view(self.num_heads,
                                                    self.embed_size,
                                                    self.context_size)

        # broadcast dot attenton
        input_expand = input.unsqueeze(
            2)  # [bs, seq_len, 1, context_size, embed]
        multi_head_pos = multi_head_pos.permute(
            0, 2, 1)  #[num_heads, context_size, embed]
        weight = input_expand * multi_head_pos  # [bs, seq_len, num_heads, context_size, embed]
        weight = weight.sum(dim=-1, keepdim=False).unsqueeze(
            3)  # [bs, seq_len, num_heads, 1, context_size]
        output = weight.matmul(input_expand).squeeze(
            dim=3)  # [bs, seq_len, num_heads, embed]
        output = output.sum(dim=2)  # [bs, seq_len, embed]
        output = output / (self.num_heads * self.context_size)

        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        return output

    def forward_step(
        self,
        input: torch.Tensor,
        padding: torch.Tensor,
        history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ forward step for inference
        Args:
            input (torch.Tensor): [batch_size, time_step=1]
            history (torch.Tensor): [batch_size, history_size, embed]
            padding (torch.Tensor): [batch_size,1], 1 is padding value

        """
        assert input.size(1) == 1
        assert history.size(1) == self.context_size - 1
        input = self.embed(input)  # [bs, 1, embed]
        input_expand = torch.cat(
            (history, input),
            dim=1).unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, context_size, embed]

        # multi head pos: [n_head, embed, context_size]
        multi_head_pos = self.pos_embed.weight.view(self.num_heads,
                                                    self.embed_size,
                                                    self.context_size)

        multi_head_pos = multi_head_pos.permute(
            0, 2, 1)  # [num_heads, context_size, embed]
        weight = input_expand * multi_head_pos  # [bs, 1, num_heads, history_size, embed]
        weight = weight.sum(dim=-1, keepdim=False).unsqueeze(
            3)  # [bs, 1, num_heads, 1, context_size]
        output = weight.matmul(input_expand).squeeze(
            dim=3)  # [bs, 1, num_heads, embed]
        output = output.sum(dim=2)  # [bs, 1, embed]
        output = output / (self.num_heads * self.context_size)

        output = self.ffn(output)
        output = self.norm(output)
        output = self.activatoin(output)
        output = ApplyPadding(input, padding, output)
        return output, input[1:, :, :]
