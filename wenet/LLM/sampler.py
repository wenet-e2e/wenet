from typing import Union
import torch


# modified from https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L26
@torch.no_grad()
def sampler(
    logits: torch.Tensor,
    temperatures: Union[torch.Tensor, None],
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
) -> torch.Tensor:
    assert logits.size(1) == 1
    logits = logits.squeeze(1)  # (batch_size, vocab_size)
    if temperatures is None:
        return torch.argmax(logits, dim=-1).squeeze(dim=-1)

    # Apply temperature scaling.
    logits.div_(temperatures.unsqueeze(dim=1))

    # Calculate probabilities with softmax.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Apply top-p, top-k.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    probs_sort = torch.where(top_ps_mask, 0, probs_sort)

    top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
    top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
    probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalization.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = torch.gather(probs_sort,
                         dim=-1,
                         index=torch.argsort(probs_idx, dim=-1))

    next_token_ids = torch.multinomial(probs, num_samples=1,
                                       replacement=True).squeeze(dim=-1)
    return next_token_ids
