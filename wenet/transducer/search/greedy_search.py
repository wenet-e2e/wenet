from typing import List

import torch
from torch import nn


def basic_greedy_search(
    model: nn.Module,
    enc_out: torch.Tensor,
    enc_out_lens: int,
    n_step: int = 100,
) -> List[List[int]]:
    """ basic greedy search

    Args:
        model (Transducer): transducer model
        enc_out (torch.Tensor): [Batch=1, T, dim]
        enc_out_lens (int): enc_out valid length
    Returns:
        List[List[int]]: best path result
    """
    # fake padding
    padding = torch.zeros(1, 1)
    # sos
    pred_input_step = torch.tensor([model.blank]).reshape(1, 1)
    state_m, state_c = model.predictor.init_state(1, method="zero")
    t = 0
    hyps = []
    prev_out_nblk = True
    pred_out_step = None
    per_frame_max_noblk = n_step
    per_frame_noblk = 0
    while t < enc_out_lens:
        encoder_out_step = enc_out[:, t:t + 1, :]  # [1, 1, E]
        if prev_out_nblk:
            pred_out_step, state_out_m, state_out_c = model.predictor.forward_step(
                pred_input_step, padding, state_m, state_c)  # [1, 1, P]

        joint_out_step = model.joint(encoder_out_step,
                                     pred_out_step)  # [1,1,v]
        joint_out_probs = joint_out_step.log_softmax(dim=-1)
        joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()  # []
        if joint_out_max != model.blank:
            hyps.append(joint_out_max.item())
            prev_out_nblk = True
            per_frame_noblk = per_frame_noblk + 1
            pred_input_step = joint_out_max.reshape(1, 1)
            state_m, state_c = state_out_m, state_out_c

        if joint_out_max == model.blank or per_frame_noblk >= per_frame_max_noblk:
            if per_frame_noblk >= per_frame_max_noblk:
                prev_out_nblk = False
            # TODO(Mddct): make t in chunk for streamming
            # or t should't be too lang to predict none blank
            t = t + 1
            per_frame_noblk = 0

    return [hyps]
