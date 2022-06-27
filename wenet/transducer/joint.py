import torch
from torch import nn
from typeguard import check_argument_types
from wenet.utils.common import get_activation


class TransducerJoint(torch.nn.Module):

    def __init__(self,
                 voca_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 prejoin_linear: bool = True,
                 postjoin_linear: bool = False,
                 joint_mode: str = 'add',
                 activation: str = "tanh"):
        assert check_argument_types()
        # TODO(Mddct): concat in future
        assert joint_mode in ['add']
        super().__init__()

        self.activatoin = get_activation(activation)
        self.prejoin_linear = prejoin_linear
        self.postjoni_linear = postjoin_linear
        self.joint_mode = joint_mode

        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)
            # just for torchscript
            self.post_ffn = nn.Linear(enc_output_size, join_dim)

        if self.postjoni_linear:
            # post join means: enc_output_size == pred_output_size
            assert enc_output_size == pred_output_size
            self.post_ffn = nn.Linear(enc_output_size, join_dim)

        self.ffn_out = nn.Linear(join_dim, voca_size)

    def forward(self, enc_out: torch.Tensor, pred_out: torch.Tensor):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if self.prejoin_linear:
            enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,V]
            pred_out = self.pred_ffn(pred_out)

        enc_out = enc_out.unsqueeze(2)  # [B,T,V] -> [B,T,1,V]
        pred_out = pred_out.unsqueeze(1)  #[B,U,V] -> [B,1 U, V]

        # TODO(Mddct): concat joint
        _ = self.joint_mode
        out = enc_out + pred_out  # [B,T,U,V]

        if self.postjoni_linear:
            out = self.post_ffn(out)

        out = self.activatoin(out)
        out = self.ffn_out(out)
        return out
