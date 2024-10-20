import math
from typing import Dict, Tuple
import torch
from wenet.ssl.bestrq.bestrq_model import quantize_vector

from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.mask import make_non_pad_mask


class NestRQModel(torch.nn.Module):
    """ https://arxiv.org/pdf/2409.08680
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        num_mel_bins: int = 80,
        embedding_dim: int = 16,
        num_embeddings: int = 8192,
        num_codebooks: int = 1,
        n_subsequent: int = 1,
        out_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.num_embeddings = num_embeddings

        # encoder
        self.encoder = encoder
        # n softmax
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.empty(n_subsequent, self.num_codebooks,
                        self.encoder.output_size(), num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.out_bias = out_bias
        if self.out_bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(
                torch.empty(n_subsequent, self.num_codebooks, num_embeddings))
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
        self.norm = torch.nn.LayerNorm(self.stack_frames * num_mel_bins,
                                       eps=1e-6,
                                       elementwise_affine=False,
                                       bias=False)
        # Section: 1B
        self.n_subsequent = n_subsequent

        # codebook
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

        # 1 stack fbank, out_mask is for compute loss (NPT)
        stack_input, stack_out_mask = self._stack_features(input, xs_lens)

        # 2 get nearest embedding
        target_ids = self._nearest_embedding_idx(stack_input)
        target_ids = target_ids[:, :stack_out_mask.size(1), :]
        target_ids = target_ids.unfold(1, size=self.n_subsequent,
                                       step=1).transpose(-1,
                                                         -2)  # (B,T,-1, vocab)

        # 3 forward xxx-formaer block and its subsampling layer
        # TODO(mddct): encoder causal mask
        out, out_mask = self.encoder(xs, xs_lens)

        # 4 get logits
        out = out.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T', dim]
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [1, n_subsequent, num_codebooks, dim, num_embeddings]
        out = torch.matmul(
            out,
            top_n_out)  # [B, n_subsequent, num_codebooks, T', num_embeddings]
        if self.out_bias:
            out = out + self.encoder_top_n_out_bias.unsqueeze(0).unsqueeze(3)

        # shift input and target for next token prediction
        out = out[:, :, :, :target_ids.size(1), :]
        target_ids = target_ids[:, 1:, :, :]
        masks = out_mask.squeeze(1) * stack_out_mask
        masks = masks[:, 1:]

        # 5 compute loss
        loss = self._compute_loss(out, target_ids, mask=masks)

        # 6 other info: num codes used in batch, unique num codes used in batch
        num_codes = masks.sum() * self.num_codebooks
        uniq_num_codes = torch.tensor(
            torch.unique(target_ids * masks.unsqueeze(2)).numel()).detach()
        ids_corr = out.argmax(dim=-1, keepdim=False).transpose(1,
                                                               2) == target_ids
        codes_acc = (ids_corr * masks.unsqueeze(2)).sum() / num_codes
        return {
            "codes_acc": codes_acc,
            "loss": loss,
            "num_codes": num_codes,
            "uniq_num_codes": uniq_num_codes,
            "th_accuracy": codes_acc,
        }

    def _stack_features(
            self, input: torch.Tensor,
            input_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = make_non_pad_mask(input_lens)
        mask_stride = mask.unfold(
            1,
            size=self.stack_frames,
            step=self.stride,
        )
        subsampline_mask, _ = torch.min(mask_stride, dim=-1)

        stack_input = input.unfold(1, size=self.stack_frames, step=self.stride)
        stack_input = stack_input.transpose(-1, -2)
        b, n, f, d = stack_input.size()
        stack_input = stack_input.reshape(b, n, f * d)

        return self.norm(stack_input), subsampline_mask

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        logits = input.contiguous().permute(
            (0, 3, 1, 2, 4)).view(-1, input.size(-1))
        mask = mask.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.n_subsequent,
                                                     self.num_codebooks)
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
