import math
from typing import Dict, Optional, Tuple
import torch

from wenet.ssl.bestrq.mask import compute_mask_indices_v2
from wenet.utils.mask import make_non_pad_mask, make_pad_mask
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.encoder_layer import ConformerEncoderLayer


def quantize_vector(latent: torch.Tensor, codebook: torch.Tensor):
    """
    Symbols in comments:
    B: batch_size.
    D: latent_dim.
    C: num_latent_classes per group
    G: num of codebook groups.

    Args:
        latent: [B, D]
        codebook: [C, G, D // G]

    Returns:
        (quantized, codes, onehot).
         - quantized: [B, D]
         - codes:     [B, G]
         - onehot:    [B, G, C]
    """

    assert len(codebook.size()) == 3
    b, d = latent.size()
    c, g, _ = codebook.size()
    assert d % g == 0

    latent = latent.reshape(b, g, d // g)

    # [B, G, C]
    # torch.transpose(codebook, [2,1,0])
    distance = (
        # [b, g, 1]
        torch.sum(latent**2, -1, keepdim=True) -
        # [b, g, c]
        2 * torch.einsum('bgd,cgd->bgc', latent, codebook) +
        # [1, g, c]
        torch.sum(codebook.permute([2, 1, 0])**2, 0, keepdim=True))

    # [B, G]
    codes = torch.argmin(distance, dim=-1)

    # [B, G, C]
    one_hot = torch.nn.functional.one_hot(codes, c).type(codebook.dtype)
    quantized = torch.einsum('bgc,cgd->bgd', one_hot, codebook)
    quantized = torch.reshape(quantized, [b, d])
    return quantized, codes, one_hot


class BestRQModel(torch.nn.Module):

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_mel_bins: int = 80,
        embedding_dim: int = 16,
        num_embeddings: int = 8192,
        num_codebooks: int = 1,
        mask_prob: float = 0.01,
        mask_length: int = 10,
        min_masks: int = 2,
        norm_epsilon: float = 1e-5,
        out_bias: bool = False,
        features_regularization_weight: float = 0.01,
    ) -> None:
        super().__init__()
        assert mask_prob > 0.0
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks

        self.num_codebooks = num_codebooks
        self.num_embeddings = num_embeddings
        self.features_regularization_weight = features_regularization_weight

        # encoder
        self.encoder = encoder
        # n softmax
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.empty(self.num_codebooks, self.encoder.output_size(),
                        num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.out_bias = out_bias
        if self.out_bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(
                torch.empty(self.num_codebooks, num_embeddings))
            torch.nn.init.zeros_(self.encoder_top_n_out_bias)

        # stack input: eg: fbank
        self.stack_frames = self.encoder.embed.right_context + 1
        self.stride = self.encoder.embed.subsampling_rate
        input_dim = num_mel_bins * self.stride

        # random projectoin
        self.projection = torch.nn.parameter.Parameter(
            torch.empty(input_dim, embedding_dim * self.num_codebooks),
            requires_grad=False,
        )
        torch.nn.init.xavier_uniform_(self.projection)

        # codebooks
        # [num_embeddings, num_codebooks, num_embeddings] means
        # [C, G, D] see quantize_vector
        self.embeddings = torch.nn.parameter.Parameter(
            torch.empty(num_embeddings, self.num_codebooks, embedding_dim),
            requires_grad=False,
        )
        torch.nn.init.normal_(self.embeddings)
        self.embeddings /= (self.embeddings.norm(dim=-1, p=2, keepdim=True) +
                            1e-8)

        # force reset encoder papameter
        self.reset_encoder_parameter()

    def reset_encoder_parameter(self):

        def _reset_parameter(module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight.data,
                                            mean=0.0,
                                            std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups /
                                  (module.in_channels * module.kernel_size[0]))
                    torch.nn.init.uniform_(module.bias, a=-k, b=k)
            elif isinstance(module, torch.Tensor):
                torch.nn.init.trunc_normal_(module)
            else:
                raise NotImplementedError("other module not support now")

        encoders = self.encoder.encoders
        for _, layer in enumerate(encoders):
            self_attn = layer.self_attn
            _reset_parameter(self_attn.linear_q)
            _reset_parameter(self_attn.linear_k)
            _reset_parameter(self_attn.linear_v)
            _reset_parameter(self_attn.linear_out)
            if isinstance(self_attn, RelPositionMultiHeadedAttention):
                _reset_parameter(self_attn.pos_bias_u)
                _reset_parameter(self_attn.pos_bias_v)
            if isinstance(layer, ConformerEncoderLayer):
                conv1, conv2 = (layer.conv_module.pointwise_conv1,
                                layer.conv_module.depthwise_conv)
                _reset_parameter(conv1)
                _reset_parameter(conv2)

    def forward(
        self,
        batch: Dict,
        device: torch.device,
    ):
        xs = batch['feats'].to(device)
        xs_lens = batch['feats_lengths'].to(device)
        input = xs

        features_pen: Optional[torch.Tensor] = None
        if self.features_regularization_weight != 0.0:
            features_pen = input.pow(2).mean()

        # 1 mask input
        xs, code_ids_mask = self._apply_mask_signal(xs, xs_lens)

        # 2.0 stack fbank
        unmasked_xs = self._stack_features(input, xs_lens)
        masked_xs = xs

        # 2.1 get nearest embedding
        target_ids = self._nearest_embedding_idx(unmasked_xs)
        target_ids = target_ids[:, :code_ids_mask.size(1), :]

        # 3 forward xxx-formaer block and its subsampling layer
        out, out_mask = self.encoder(masked_xs, xs_lens)

        # 4 get logits
        out = out.unsqueeze(1)  # [B, 1, T', dim]
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [1, num_codebooks, dim, num_embeddings]
        out = torch.matmul(out,
                           top_n_out)  # [B, num_codebooks, T', num_embeddings]
        if self.out_bias:
            out = out + self.encoder_top_n_out_bias.unsqueeze(0).unsqueeze(2)

        # 5 compute loss
        masks = out_mask.squeeze(1) * code_ids_mask
        loss = self._compute_loss(out, target_ids, mask=masks)
        if self.features_regularization_weight != 0.0:
            loss = loss + self.features_regularization_weight * features_pen

        # 6 other info: num codes used in batch, unique num codes used in batch
        num_codes = masks.sum() * self.num_codebooks
        uniq_num_codes = torch.tensor(
            torch.unique(target_ids * masks.unsqueeze(2)).numel()).detach()
        ids_corr = out.argmax(dim=-1, keepdim=False).transpose(1,
                                                               2) == target_ids
        codes_acc = (ids_corr * masks.unsqueeze(2)).sum() / num_codes
        return {
            "codes_acc": codes_acc,
            "features_l2": features_pen,
            "loss": loss,
            "num_codes": num_codes,
            "uniq_num_codes": uniq_num_codes,
            "th_accuracy": codes_acc,
        }

    def _apply_mask_signal(
            self, input: torch.Tensor,
            input_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        B, T, _ = input.size()
        padding_mask = make_pad_mask(input_lens)

        # calc subsampling masks
        padding_mask_stride = padding_mask.unfold(
            1,
            size=self.stack_frames,
            step=self.stride,
        )
        padding_mask, _ = torch.max(padding_mask_stride, dim=-1)
        masks = compute_mask_indices_v2(padding_mask.size(),
                                        padding_mask,
                                        self.mask_prob,
                                        self.mask_length,
                                        min_masks=self.min_masks,
                                        device=device)
        # calc signal mask
        subsampling_mask = masks
        bool_stride_mask = torch.ones_like(padding_mask_stride, device=device)
        mask_stride = torch.where(masks.unsqueeze(-1), bool_stride_mask, False)
        # recover orign seq masks
        masks = mask_stride[:, :, :self.stride].flatten(start_dim=1)
        masks_padding = torch.zeros(
            B,
            T,
            device=device,
            dtype=padding_mask.dtype,
        )
        masks_padding[:, :masks.size(-1)] = masks
        masks = masks_padding
        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]
        # NOTE(Mddct): you can use size (b,t,d) for torch.normal
        mask_emb = torch.normal(mean=0, std=0.1,
                                size=(1, 1, input.size(2))).to(input.device)
        xs = torch.where(masks_expand, mask_emb, input)
        return xs, subsampling_mask

    def _stack_features(self, input: torch.Tensor,
                        input_lens: torch.Tensor) -> torch.Tensor:

        stack_input = input.unfold(1, size=self.stride, step=self.stride)
        stack_input = stack_input.transpose(-1, -2)
        b, n, f, d = stack_input.size()
        stack_input = stack_input.reshape(b, n, f * d)

        # NOTE(Mddct): important!!!
        # norm stack features
        mask = make_non_pad_mask(input_lens)
        stack_mask = mask.unfold(1, size=self.stride, step=self.stride)
        stack_mask, _ = torch.min(stack_mask, dim=-1)

        stack_input = stack_input * stack_mask.unsqueeze(2)
        mean = stack_input.sum(1, keepdim=True) / stack_mask.sum(
            dim=1, keepdim=True).unsqueeze(1)
        std = torch.sqrt(((stack_input - mean)**2).sum(dim=1, keepdim=True) /
                         stack_mask.sum(dim=1, keepdim=True).unsqueeze(1))
        norm_stack_input = (stack_input - mean) / (std + 1e-5)
        return norm_stack_input

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        logits = input.transpose(1, 2).contiguous().view(-1, input.size(-1))
        loss = torch.nn.functional.cross_entropy(
            logits,
            target.contiguous().view(-1),
            reduction='none',
        )
        loss = (loss * mask.view(-1)).sum() / mask.sum()
        return loss

    def _nearest_embedding_idx(self, xs: torch.Tensor) -> torch.Tensor:
        xs = torch.matmul(xs, self.projection.to(xs.device))
        xs = xs / (xs.norm(dim=-1, p=2, keepdim=True) + 1e-8)
        codebooks = self.embeddings
        B, T, C = xs.size()
        xs_flatten = xs.view(B * T, C)
        _, codes, _ = quantize_vector(xs_flatten, codebooks)
        return codes.reshape(B, T, -1)  # [B, T, num_codebooks]
