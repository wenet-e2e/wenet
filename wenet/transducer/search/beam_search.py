import copy
import heapq
from typing import List

import torch
from torch import nn
from wenet.transducer.search.hyps import Hyp


def basic_beam_search(
    model: nn.Module,
    enc_out: torch.Tensor,
    enc_out_lens: int,
    beam_size: int,
    n_step: int = 1,
    first_beam: int = 16,
) -> List[List[int]]:
    """ basic beam search

    Args:
        model (Transducer): transducer model
        enc_out (torch.Tensor): [Batch=1, T, dim]
        enc_out_lens (int): enc_out valid length
    Returns:
        List[List[int]]: best path result
    """
    assert first_beam >= beam_size
    with torch.no_grad():

        padding = torch.zeros(1, 1)
        state_m, state_c = model.predictor.init_state(1, method="zero")
        cur_hyps: List[Hyp] = []
        new_prefix_hyps: List[Hyp] = []
        blank_hyp = Hyp(([0], 0.0, state_m, state_c))
        heapq.heappush(cur_hyps, blank_hyp)

        t = 0
        while t < enc_out_lens:
            encoder_out_step = enc_out[:, t:t + 1, :]  # [1, 1, E]
            #  new_prefix beam from cur_prefx
            new_prefix_hyps_tmp = []
            for i in range(beam_size):
                if len(cur_hyps) == 0:
                    break
                heapq.heappush(new_prefix_hyps_tmp, heapq.heappop(cur_hyps))
            new_prefix_hyps = new_prefix_hyps_tmp
            cur_hyps = []

            t = t + 1
            u = 0
            while u <= n_step:
                # get max from new_prefix_hyps and remove
                cur_max_prefix = heapq.heappop(new_prefix_hyps)
                state_m, state_c = cur_max_prefix.state()
                ids = torch.tensor(cur_max_prefix.end()).reshape(1, 1)
                pred_out_step, state_out_m, state_out_c = model.predictor.forward_step(
                    ids, padding, state_m, state_c)
                joint_out_step = model.joint(encoder_out_step, pred_out_step)
                joint_out_probs = joint_out_step.log_softmax(dim=-1)  # [1,1,v]
                joint_out_probs: torch.Tensor = joint_out_probs.squeeze(
                )  # (v,)

                # non blank topk
                # first prune
                topk_v, topk_i = joint_out_probs[1:].topk(first_beam)
                for i in range(first_beam):
                    prefix = copy.deepcopy(cur_max_prefix)
                    prefix.push(
                        int(topk_i[i].item()) + 1, topk_v[i].item(),
                        state_out_m, state_out_c)
                    heapq.heappush(new_prefix_hyps, prefix)

                # blank
                cur_max_prefix.push(0, joint_out_probs[0].item(), state_m,
                                    state_c)
                heapq.heappush(cur_hyps, cur_max_prefix)

                if len(cur_hyps) >= beam_size:
                    topk_gt = False
                    sorted(cur_hyps, key=lambda hyp: hyp.logp, reverse=True)
                    if cur_hyps[-1].logp >= new_prefix_hyps[0].logp:
                        topk_gt = True
                    if topk_gt:
                        cur_hyps = cur_hyps[:beam_size]
                        break

                u = u + 1

        # now cur_hyps is topk seq
        nbest: List[List[int]] = []
        if len(cur_hyps) != 0:
            new_prefix_hyps = []
            for i in range(beam_size):
                if len(cur_hyps) == 0:
                    break
                heapq.heappush(new_prefix_hyps, heapq.heappop(cur_hyps))

        for hyp in sorted(new_prefix_hyps,
                          key=lambda hyp: hyp.logp,
                          reverse=True):
            if len(hyp.prefix[1:]) != 0:
                nbest.append(hyp.prefix[1:])  # remove blank
        return nbest
