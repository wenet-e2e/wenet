import math
from typing import Dict, Optional, Tuple
import torch

from wenet.ssl.bestrq.mask import compute_mask_indices_v2
from wenet.utils.mask import make_pad_mask
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
        assert self.encoder.global_cmvn is not None
        self.register_buffer('signal_mean', self.encoder.global_cmvn.mean)
        self.register_buffer('signal_istd', self.encoder.global_cmvn.istd)
        self.signal_norm_var = self.encoder.global_cmvn.norm_var
        # NOTE(Mddct): disable encoder's global_cmvn
        self.encoder.global_cmvn = None

        # n softmax
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.empty(self.num_codebooks, self.encoder.output_size(),
                        num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(
            torch.empty(self.num_codebooks, num_embeddings))
        torch.nn.init.zeros_(self.encoder_top_n_out_bias)

        # stack input: eg: fbank
        self.stack_frames = self.encoder.embed.right_context + 1
        self.stride = self.encoder.embed.subsampling_rate
        input_dim = num_mel_bins * self.stack_frames

        # norm input
        self.norm = torch.nn.LayerNorm(
            input_dim, eps=norm_epsilon, elementwise_affine=False
        ) if self.stack_frames > 1 else torch.nn.Identity()

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
        # force global cmvn
        xs = xs - self.signal_mean
        if self.signal_norm_var:
            xs = xs * self.signal_istd
        input = xs

        features_pen: Optional[torch.Tensor] = None
        if self.features_regularization_weight != 0.0:
            features_pen = input.pow(2).mean()

        # 0 mask input
        xs, masked_masks = self._apply_mask_signal(xs, xs_lens)

        # 1 get subsampling mask
        subsampling_masks = masked_masks.unfold(1,
                                                size=self.stack_frames,
                                                step=self.stride)
        code_ids_mask, _ = torch.min(subsampling_masks, 2)

        # 2.0 stack fbank
        unmasked_xs = self._stack_features(input)
        masked_xs = xs

        # 2.1 get nearest embedding
        target_ids = self._nearest_embedding_idx(unmasked_xs)

        # 3 forward xxx-formaer block and its subsampling layer
        out, out_mask = self.encoder(masked_xs, xs_lens)

        # 4 get logits
        out = out.unsqueeze(1)  # [B, 1, T', dim]
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [1, num_codebooks, dim, num_embeddings]
        out = torch.matmul(out,
                           top_n_out)  # [B, num_codebooks, T', num_embeddings]
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
            "uniq_num_codes": uniq_num_codes
        }

    def _apply_mask_signal(
            self, input: torch.Tensor,
            input_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_mask = make_pad_mask(input_lens)
        masks = compute_mask_indices_v2(input.size()[:-1],
                                        padding_mask,
                                        self.mask_prob,
                                        self.mask_length,
                                        min_masks=self.min_masks,
                                        device=input.device)

        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]
        mask_emb = torch.normal(mean=0, std=0.1,
                                size=(1, 1, input.size(2))).to(input.device)
        xs = torch.where(masks_expand, mask_emb, input)
        return xs, masks

    def _stack_features(self, input: torch.Tensor) -> torch.Tensor:

        stack_input = input.unfold(1, size=self.stack_frames, step=self.stride)
        stack_input = stack_input.transpose(-1, -2)
        b, n, f, d = stack_input.size()
        stack_input = stack_input.reshape(b, n, f * d)

        return stack_input

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(input, dim=-1).transpose(
            1, 2)  # [B, T', num_codebooks, num_embeddings]

        per_example_n_loss = -log_probs.gather(3, target.unsqueeze(3)).squeeze(
            3)  # [B, T', num_codebooks]

        numerator = torch.sum(per_example_n_loss * mask.unsqueeze(2))
        denominator = torch.sum(mask) + 1e-5
        loss = numerator / (denominator * self.num_codebooks)
        return loss

    def _nearest_embedding_idx(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.norm(xs)
        xs = torch.matmul(xs, self.projection.to(xs.device))

        B, T, C = xs.size()
        xs_flatten = xs.view(B * T, C)
        _, codes, _ = quantize_vector(xs_flatten, self.embeddings)
        return codes.reshape(B, T, -1)  # [B, T, num_codebooks]
