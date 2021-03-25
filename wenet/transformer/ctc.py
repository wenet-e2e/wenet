import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from wenet.utils.common import insert_blank


class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)

    def forced_align(self,
                     ctc_probs: torch.Tensor,
                     y: torch.Tensor,
                     blank_id=0) -> list:
        """ctc forced alignment.

        Args:
            torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
            torch.Tensor y: id sequence tensor 1d tensor (L)
            int blank_id: blank symbol index
        Returns:
            torch.Tensor: alignment result
        """

        ctc_hyps = ctc_probs

        y_insert_blank = insert_blank(y, blank_id)

        log_alpha = torch.zeros((ctc_hyps.size(0), len(y_insert_blank)))
        log_alpha = log_alpha - float('inf')  # log of zero
        state_path = (torch.zeros(
            (ctc_hyps.size(0), len(y_insert_blank)), dtype=torch.int16) - 1
        )  # state path

        # init start state
        log_alpha[0, 0] = ctc_hyps[0][y_insert_blank[0]]
        log_alpha[0, 1] = ctc_hyps[0][y_insert_blank[1]]

        for t in range(1, ctc_hyps.size(0)):
            for s in range(len(y_insert_blank)):
                if y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[
                        s] == y_insert_blank[s - 2]:
                    candidates = torch.tensor(
                        [log_alpha[t - 1, s], log_alpha[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = torch.tensor([
                        log_alpha[t - 1, s],
                        log_alpha[t - 1, s - 1],
                        log_alpha[t - 1, s - 2],
                    ])
                    prev_state = [s, s - 1, s - 2]
                log_alpha[
                    t,
                    s] = torch.max(candidates) + ctc_hyps[t][y_insert_blank[s]]
                state_path[t, s] = prev_state[torch.argmax(candidates)]

        state_seq = -1 * torch.ones((ctc_hyps.size(0), 1), dtype=torch.int16)

        candidates = torch.tensor([
            log_alpha[-1, len(y_insert_blank) - 1],
            log_alpha[-1, len(y_insert_blank) - 2]
        ])
        prev_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
        state_seq[-1] = prev_state[torch.argmax(candidates)]
        for t in range(ctc_hyps.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_alignment = []
        for t in range(0, ctc_hyps.size(0)):
            output_alignment.append(y_insert_blank[state_seq[t, 0]])

        return output_alignment
