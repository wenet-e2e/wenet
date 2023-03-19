from typing import Optional, Tuple
import torch

from wenet.ssl.bestrq.mask import compute_mask_indices
from wenet.utils.mask import make_pad_mask


class BestRQModel(torch.nn.Module):

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_mel_bins: int = 80,
        input_dim: int = 256,
        embedding_dim: int = 16,
        num_embeddings: int = 8192,
        num_codebooks: int = 1,
        mask_prob: float = 0.01,
        mask_length: int = 10,
        min_masks: int = 2,
        norm: str = "batch_norm",
        norm_epsilon: float = 1e-5,
        mask_signal: bool = False,
    ) -> None:
        super().__init__()

        assert mask_prob > 0.0

        self.mask_prob = mask_prob
        # NOTE: should filter audio less than mask_length
        self.mask_length = mask_length
        self.min_masks = min_masks
        # encoder
        self.encoder = encoder
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.Tensor(num_codebooks, self.encoder.output_size(),
                         num_embeddings))
        # mask embedding
        mask_embedding_dim = num_mel_bins if mask_signal else input_dim
        self.mask_emb = torch.nn.parameter.Parameter(
            torch.Tensor(mask_embedding_dim).normal_(mean=0, std=0.1))

        # stack feature or not
        self.mask_signal = mask_signal
        if self.mask_signal:
            self.stack_frames = self.encoder.embed.subsampling_rate
            input_dim = num_mel_bins * self.stack_frames

        # norm input
        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = torch.nn.BatchNorm1d(input_dim, norm_epsilon)
        else:
            self.use_layer_norm = True
            self.norm = torch.nn.LayerNorm(input_dim, norm_epsilon)

        # random projectoin
        random_projection_weight = torch.empty(input_dim,
                                               embedding_dim,
                                               requires_grad=False)
        self.projection = torch.nn.init.xavier_normal_(
            random_projection_weight)

        # codebooks
        # [num_codebooks, embedding_dim, num_embeddings]
        random_embedding_weight = torch.empty(num_codebooks,
                                              embedding_dim,
                                              num_embeddings,
                                              requires_grad=False)
        self.embeddings = torch.nn.init.normal_(random_embedding_weight)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        text_length: Optional[torch.Tensor] = None,
    ):
        input = xs
        # should support nonstreamming and streamming
        # TODO(Mddct): streamming future
        # eg: full attenton and chunk or  dynamic chunk training

        # 0 mask signal or not
        if self.mask_signal:
            xs, masked_masks = self._apply_mask_signal(xs)
        else:
            masked_masks = None

        # 1 forward subsampling
        xs, pos_emb, masks = self._forward_subsampling(xs, xs_lens)
        if not self.mask_signal:
            unmasked_xs = xs
        else:
            assert masked_masks is not None
            unmasked_xs = self._stack_features(input, masks)

        # 2 mask subsampling features
        # 2.0 apply mask
        if not self.mask_signal:
            masked_xs, masked_masks = self._apply_mask(xs)
        else:
            masked_xs = xs

        masked_masks = masked_masks[:, :masks.size(2)]
        # 2.1 get nearest embedding
        target_ids = self._nearest_embedding_idx(unmasked_xs)

        # 3 forward xxx-formaer block
        out, out_mask = self._forward_encoder_blocks(masked_xs, masks, pos_emb,
                                                     masks)
        # 4 get logits
        out = out.unsqueeze(1)  # [B, 1, T', dim]
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [1, num_codebooks, dim, num_embeddings]
        out = torch.matmul(out,
                           top_n_out)  # [B, num_codebooks, T', num_embeddings]

        # 5 compute loss
        loss = self._compute_loss(out, target_ids,
                                  out_mask.squeeze(1) * masked_masks)
        return {"loss": loss}

    def _apply_mask_signal(
            self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = input.size()
        T_subsampling = T // self.stack_frames
        # [B, T_subsampling]
        masks = compute_mask_indices(torch.Size((B, T_subsampling)),
                                     self.mask_prob,
                                     self.mask_length,
                                     self.min_masks,
                                     device=input.device)
        zeros = torch.zeros(B,
                            T - T_subsampling * self.stack_frames,
                            dtype=masks.dtype,
                            device=input.device)
        masks_upsample = masks.unsqueeze(2).repeat(1, 1, self.stack_frames)
        masks_upsample = masks_upsample.view(-1,
                                             T_subsampling * self.stack_frames)
        signal_masks = torch.cat([masks_upsample, zeros], dim=1)  # [B, T]
        mask_emb = self.mask_emb.to(input.device).view(1, 1, -1)

        return torch.where(signal_masks.unsqueeze(2), mask_emb, input), masks

    def _stack_features(self, input: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        max_length = mask.size(2)
        stack_frames = self.stack_frames
        stack_input = input.unfold(1, stack_frames, stack_frames)
        b, n, f, d = stack_input.size()
        stack_input = stack_input.reshape(b, n, f * d)
        stack_input = stack_input[:, :max_length, :]

        return stack_input * mask.transpose(1, 2)

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        input = input.transpose(1, 3)  # [B, num_embeddings, T' num_codebooks]
        entropy = torch.nn.functional.cross_entropy(
            input, target, reduction='none')  # [B, T', num_codebooks]
        # stop gradient for non mask area
        loss = entropy * mask.unsqueeze(2)
        return loss.sum() / (mask.sum() * loss.size(2))

    def _forward_encoder_blocks(
            self, xs: torch.Tensor, xs_masks: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        masks = xs_masks
        for layer in self.encoder.encoders:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def _nearest_embedding_idx(self, xs: torch.Tensor) -> torch.Tensor:
        if not self.use_layer_norm:
            xs = xs.transpose(1, 2)
        xs = self.norm(xs)
        if not self.use_layer_norm:
            xs = xs.transpose(1, 2)
        xs = torch.matmul(xs, self.projection.to(xs.device))

        B, T, C = xs.size()
        flattened_input = xs.view(-1, C)
        embeddings = self.embeddings.to(
            xs.device)  # [num_codebooks, embedding_dim, num_embeddings]
        # [num_codebooks, B*T, num_embeddings]
        distance = (
            torch.sum(flattened_input**2, dim=1, keepdim=True).unsqueeze(0) +
            torch.sum(embeddings**2, dim=1, keepdim=True) -
            2 * torch.matmul(flattened_input.unsqueeze(0), embeddings))

        out = torch.argmin(distance, dim=-1)  # [num_codebooks, B*T]
        out = out.transpose(0, 1)  # [B*T, num_codebooks]
        return out.reshape(B, T, -1)  # [B, T, num_codebooks]

    def _apply_mask(self,
                    xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = compute_mask_indices(xs.size()[:-1],
                                     self.mask_prob,
                                     self.mask_length,
                                     self.min_masks,
                                     device=xs.device)
        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]
        mask_emb = self.mask_emb.to(xs.device).view(1, 1, -1)
        xs = torch.where(masks_expand, mask_emb, xs)
        return xs, masks

    def _forward_subsampling(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks
