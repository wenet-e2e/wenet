from typing import Tuple
import torch


def gumbel(shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """Sample Gumbel random values with given shape and float dtype.

    The values are distributed according to the probability density function:

    .. math::
     f(x) = e^{-(x + e^{-x})}

    Args:
      shape (torch.Size): pdf shape
      dtype (torch.dtype): pdf value dtype

    Returns:
       A random array with the specified shape and dtype.
    """
    # see https://www.cnblogs.com/initial-h/p/9468974.html for more details
    return -torch.log(-torch.log(
        torch.empty(shape, device=device).uniform_(
            torch.finfo(dtype).tiny, 1.)))


class Wav2vecGumbelVectorQuantizer(torch.nn.Module):

    def __init__(self,
                 features_dim: int = 256,
                 num_codebooks: int = 2,
                 num_embeddings: int = 8192,
                 embedding_dim: int = 16,
                 hard: bool = False) -> None:

        super().__init__()

        self.num_groups = num_codebooks
        self.num_codevectors_per_group = num_embeddings
        # codebooks
        # means [C, G, D] see quantize_vector in bestrq_model.py
        assert embedding_dim % num_codebooks == 0.0
        self.embeddings = torch.nn.parameter.Parameter(
            torch.empty(1, num_codebooks * num_embeddings,
                        embedding_dim // num_codebooks),
            requires_grad=True,
        )
        torch.nn.init.uniform_(self.embeddings)

        self.weight_proj = torch.nn.Linear(features_dim,
                                           num_codebooks * num_embeddings)
        # use gumbel softmax or argmax(non-differentiable)
        self.hard = hard

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:

            mask_extended = torch.broadcast_to(mask.flatten()[:, None, None],
                                               probs.shape)
            probs = torch.where(mask_extended.to(torch.bool), probs,
                                torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(
            marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(
        self,
        input: torch.Tensor,
        input_mask: torch.Tensor,
        temperature: float = 1.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        b, t, _ = input.size()

        hidden = self.weight_proj(input)
        hidden = hidden.reshape(b * t * self.num_groups, -1)
        if not self.hard:
            # sample code vector probs via gumbel in differentiateable way
            gumbels = gumbel(hidden.size(), hidden.dtype, hidden.device)
            codevector_probs = torch.nn.functional.softmax(
                (hidden + gumbels) / temperature, dim=-1)

            # compute perplexity
            codevector_soft_dist = torch.nn.functional.softmax(
                hidden.reshape(b * t, self.num_groups, -1),
                dim=-1,
            )  # [B*T, num_codebooks, num_embeddings]
            perplexity = self._compute_perplexity(codevector_soft_dist,
                                                  input_mask)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden.argmax(axis=-1)
            codevector_probs = torch.nn.functional.one_hot(
                codevector_idx, hidden.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(
                b * t, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, input_mask)

        targets_idx = codevector_probs.argmax(-1).reshape(b, t, -1)
        codevector_probs = codevector_probs.reshape(b * t, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(
            -1) * self.embeddings
        codevectors = codevectors_per_group.reshape(
            b * t, self.num_groups, self.num_codevectors_per_group, -1)

        codevectors = codevectors.sum(-2).reshape(b, t, -1)
        return codevectors, perplexity, targets_idx
