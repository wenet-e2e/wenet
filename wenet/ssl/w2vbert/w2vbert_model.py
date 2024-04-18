import math
from typing import Dict, Optional, Tuple, Union
import torch

from wenet.ssl.bestrq.mask import compute_mask_indices_v2
from wenet.ssl.wav2vec2.quantizer import Wav2vecGumbelVectorQuantizer
from wenet.ssl.wav2vec2.wav2vec2_model import (_compute_contrastive_loss,
                                               _sample_negative_indices)
from wenet.transformer.attention import RelPositionMultiHeadedAttention

from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.mask import make_non_pad_mask


class W2VBERTModel(torch.nn.Module):

    def __init__(
        self,
        encoder: Union[ConformerEncoder, TransformerEncoder],
        embedding_dim: int = 256,
        num_embeddings: int = 320,
        num_codebooks: int = 1,
        mask_prob: float = 0.065,
        mask_length: int = 10,
        min_masks: int = 2,
        num_negatives: int = 100,
        features_regularization_weight: float = 0.01,
        max_gumbel_temperature: float = 2.0,
        min_gumbel_temperature: float = 0.1,
        gumbel_temperature_decay: float = 0.999995,
        contrastive_logits_temperature: float = 0.1,
        diversity_weight: float = 0.0,
        bias: bool = True,
        contrastive_blocks: int = 6,
        masked_blocks: int = 6,
        contrastive_weight: float = 1.0,
        mlm_weight: float = 1.0,
        warmup_steps: int = 25000,
    ) -> None:
        """ Wrap encoder to train using W2V-BERT's style

        Described in:
        https://arxiv.org/pdf/2108.06209v2.pdf

        Args:
            encoder: wenet's encoder,
                     only support conformer and transformer now
            embedding_dim: codebooks embedding dim
            num_embeddings: numbers of each codebook
            num_codebooks: numbers of codebooks i.e groups of codebook
            mask_prob: probs of mask
            mask_length: spans of masks
            min_masks: min masks for each audio
            num_negatives: numbers of negatives of each masks
            features_regularization_weight: l2 regularization weight
            max_gumbel_temperature: maximum temperature for gumbel softmax
            min_gumbel_temperature: minimum temperature for gumbel softmax
            gumbel_temperature_decay:
                decay of gumbel temperature during training
            contrastive_logits_temperature:
                the temperature in the contrastive loss.
        """
        super().__init__()
        assert mask_prob > 0.0
        assert (contrastive_blocks > 0 and masked_blocks > 0 and
                contrastive_blocks + masked_blocks == len(encoder.encoders))
        self.contrastive_blocks = contrastive_blocks
        self.masked_blocks = masked_blocks

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_negatives = num_negatives

        self.features_regularization_weight = features_regularization_weight
        self.diversity_weight = diversity_weight

        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight
        self.warmup_steps = warmup_steps
        # encoder
        self.encoder = encoder

        # quantizer
        self.num_codebooks = num_codebooks
        self.quantizer = Wav2vecGumbelVectorQuantizer(
            self.encoder.output_size(),
            num_codebooks=num_codebooks,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hard=False,
        )
        self.max_gumbel_temp = max_gumbel_temperature
        self.min_gumbel_temp = min_gumbel_temperature
        self.gumbel_temp_decay = gumbel_temperature_decay

        self.num_codevectors_per_group = num_embeddings
        self.num_codevector_groups = num_codebooks

        self.contrastive_logits_temp = contrastive_logits_temperature

        # NOET(Mddct): mask_em is replaced by random value in Wav-BERT
        # self.mask_emb = torch.nn.parameter.Parameter(
        #     torch.empty(self.encoder.output_size()).uniform_(),
        #     requires_grad=True,
        # )
        # TODO(Mddct): support causal or lookahead mask or keep consistent with
        # wenet dynamic chunk training

        # # n softmax
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.empty(num_codebooks, self.encoder.output_size(),
                        num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.bias = bias
        if bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(
                torch.empty(num_codebooks, num_embeddings))
            torch.nn.init.zeros_(self.encoder_top_n_out_bias)

        # reset parameter
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

    @torch.jit.unused
    def forward(
        self,
        batch: Dict,
        device: torch.device,
    ):
        steps = batch.get('steps', None)
        xs = batch['feats'].to(device)
        xs_lens = batch['feats_lengths'].to(device)
        assert xs.size(0) == xs_lens.size(0)
        assert steps is not None

        # 1 forward subsampling
        # NOTE(Mddct): use subsampling as feature extraction
        xs, pos_emb, masks = self._forward_subsampling(xs, xs_lens)
        unmasked_xs = xs
        # 2 mask features
        masked_xs, masked_masks = self._apply_mask(xs, masks.squeeze(1))
        # 3 forward encoder blocks
        contrastive_vec, mlm_vec, out_mask = self._forward_encoder_blocks(
            masked_xs, masks, pos_emb, masks)

        # 4 constrastive branch
        gumbel_temperature = max(
            self.max_gumbel_temp * self.gumbel_temp_decay**steps,
            self.min_gumbel_temp)

        quantized_features, codevector_perplexity, targets_ids = self.quantizer(
            unmasked_xs, masks.squeeze(1), gumbel_temperature)

        sampled_negative_indices = _sample_negative_indices(
            xs.size()[:-1], self.num_negatives, masked_masks.device,
            masked_masks)

        loss_contrastive = _compute_contrastive_loss(
            quantized_features, contrastive_vec, sampled_negative_indices,
            masked_masks, self.contrastive_logits_temp, self.num_negatives)
        loss = loss_contrastive

        # scale by sample size
        # make sure that diversity loss is multiplied by `sample_size`
        # since contrastive_loss is `sum`-reduced instead of averaged
        sample_size = masked_masks.sum()
        # higher codevector_perplexity leads to lower diversity loss
        loss_diversity: Optional[torch.Tensor] = None
        if self.diversity_weight != 0.0:
            loss_diversity = (
                self.num_codevector_groups * self.num_codevectors_per_group -
                codevector_perplexity) / (self.num_codevectors_per_group *
                                          self.num_codevector_groups)
            loss_diversity = loss_diversity * sample_size
            loss = loss + self.diversity_weight * loss_diversity
        loss = loss / sample_size

        features_pen: Optional[torch.Tensor] = None
        if self.features_regularization_weight != 0.0:
            features_pen = xs.pow(2).mean()
            loss = loss + self.features_regularization_weight * features_pen

        # 5 maked lm branch
        out = mlm_vec.unsqueeze(1)
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [1, num_codebooks, dim, num_embeddings]
        out = torch.matmul(out,
                           top_n_out)  # [B, num_codebooks, T', num_embeddings]
        if self.bias:
            out = out + self.encoder_top_n_out_bias.unsqueeze(0).unsqueeze(2)
        num_codes = masked_masks.sum() * self.num_codebooks
        loss_mlm = self._compute_mlm_loss(out,
                                          targets_ids,
                                          mask=out_mask.squeeze(1) *
                                          masked_masks)
        ids_corr = out.argmax(dim=-1,
                              keepdim=False).transpose(1, 2) == targets_ids
        codes_acc = (ids_corr * masked_masks.unsqueeze(2)).sum() / num_codes
        # TODO(Mddct): support num codes used in batch, unique num codes
        # used in batch like bestrq

        # 6 final loss
        mlm_weight = (self.mlm_weight if steps >= self.warmup_steps else 0.1 +
                      0.9 * (steps / self.warmup_steps))
        loss = self.contrastive_weight * loss + mlm_weight * loss_mlm
        return {
            "code_ppl": codevector_perplexity.detach(),
            "features_l2": features_pen,
            "codes_acc": codes_acc.detach(),
            "loss": loss,
            "loss_contrastive": loss_contrastive / sample_size,
            "loss_diversity": loss_diversity,
            "loss_mlm": loss_mlm,
        }

    def _apply_mask(
            self, xs: torch.Tensor,
            xs_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        masks = compute_mask_indices_v2(xs.size()[:-1],
                                        ~xs_masks,
                                        self.mask_prob,
                                        self.mask_length,
                                        min_masks=self.min_masks,
                                        device=xs.device)
        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]

        mask_emb = torch.normal(mean=0,
                                std=0.1,
                                size=xs.size(),
                                device=xs.device)
        xs = torch.where(masks_expand, mask_emb, xs)

        return xs, masks

    def _compute_mlm_loss(self, input: torch.Tensor, target: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(input, dim=-1).transpose(
            1, 2)  # [B, T', num_codebooks, num_embeddings]

        per_example_n_loss = -log_probs.gather(3, target.unsqueeze(3)).squeeze(
            3)  # [B, T', num_codebooks]

        numerator = torch.sum(per_example_n_loss * mask.unsqueeze(2))
        denominator = torch.sum(mask) + 1e-5
        loss = numerator / (denominator * self.num_codebooks)
        return loss

    def _forward_subsampling(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def _forward_encoder_blocks(
        self, xs: torch.Tensor, xs_masks: torch.Tensor, pos_emb: torch.Tensor,
        mask_pad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        masks = xs_masks

        xs: torch.Tensor
        # forward contrastive layers get context vector for Contrastive Loss
        for layer in self.encoder.encoders[:self.contrastive_blocks]:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        contrastive_vec = xs

        for layer in self.encoder.encoders[self.contrastive_blocks:]:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        masked_vec = xs

        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
            masked_vec = xs
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return contrastive_vec, masked_vec, masks
