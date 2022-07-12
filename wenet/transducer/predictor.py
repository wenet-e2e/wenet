from typing import Optional, Tuple

import torch
from torch import nn
from typeguard import check_argument_types


# TODO(Mddct): move to utils
def get_rnn(rnn_type: str) -> nn.Module:
    assert rnn_type in ["rnn", "lstm", "gru", "embedding"]
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
        if padding is not None:
            m = ApplyPadding(m, padding, state_m)
            c = ApplyPadding(c, padding, state_c)
        return out, m, c


class EmbeddingPredictor(torch.nn.Module):

    def __init__(self) -> None:
        assert check_argument_types()
        super().__init__()
        pass

    def forward(self):
        raise NotImplementedError("Embedding Predictor not support now")

    def forwar_step(self):
        raise NotImplementedError("Embedding Predictor not support now")
