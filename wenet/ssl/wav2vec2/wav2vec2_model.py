import math
from typing import Dict, Optional, Tuple, Union
import torch

import torch.nn.functional as F
from wenet.ssl.bestrq.mask import compute_mask_indices_v2
from wenet.ssl.wav2vec2.quantizer import Wav2vecGumbelVectorQuantizer
from wenet.transformer.attention import RelPositionMultiHeadedAttention

from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.mask import make_non_pad_mask


def _sample_negative_indices(features_shape: Tuple,
                             num_negatives: int,
                             device: torch.device,
                             mask_time_indices: Optional[torch.Tensor] = None):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    sequence_length_range = torch.arange(sequence_length, device=device)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = torch.zeros(
        (batch_size, sequence_length, num_negatives),
        dtype=sequence_length_range.dtype,
        device=device)

    mask_time_indices = (mask_time_indices.bool()
                         if mask_time_indices is not None else torch.ones(
                             features_shape, dtype=torch.bool, device=device))

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[
            mask_time_indices[batch_idx]]

        feature_indices = torch.arange(high + 1).unsqueeze(1).expand(
            high + 1, num_negatives)
        sampled_indices = torch.randint(0,
                                        high,
                                        size=(high + 1, num_negatives))
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[
            batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices.reshape(batch_size, -1)


def _compute_contrastive_loss(quantized_features: torch.Tensor,
                              features: torch.Tensor,
                              negative_indices: torch.Tensor,
                              mask_time_indices: torch.Tensor,
                              logits_temp: float,
                              num_negatives: int = 1):
    batch_size, sequence_length, hidden_size = quantized_features.shape

    # take negative vectors from sampled indices
    quantized_negatives = quantized_features.view(
        -1, hidden_size)[negative_indices.view(-1)]
    quantized_negatives = quantized_negatives.view(batch_size, sequence_length,
                                                   num_negatives,
                                                   hidden_size).permute(
                                                       2, 0, 1, 3)

    target_features = torch.cat(
        [quantized_features.unsqueeze(0), quantized_negatives], dim=0)
    loss_logits = F.cosine_similarity(features, target_features, dim=-1)
    loss_logits = loss_logits / logits_temp

    neg_is_pos = (quantized_features == quantized_negatives).all(-1)
    neg_is_pos = torch.cat(
        [
            torch.full(
                (1, ) + loss_logits.shape[1:], False,
                device=neg_is_pos.device), neg_is_pos
        ],
        dim=0,
    )

    # make sure incorrectly sampled vectors don't contribute to loss
    loss_logits = torch.where(neg_is_pos, -1e9, loss_logits)

    predictions = loss_logits.permute(2, 1, 0).reshape(-1,
                                                       loss_logits.shape[0])
    targets = ((1 - mask_time_indices.long()) * -100).transpose(1, 0).flatten()

    target_mask = torch.where(targets >= 0, 1.0, 0.0)
    contrastive_loss = F.cross_entropy(
        predictions, targets.long(), reduction='none') * target_mask

    contrastive_loss = contrastive_loss.sum()

    return contrastive_loss


class Wav2vec2Model(torch.nn.Module):

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
    ) -> None:
        """ Wrap encoder to train using wav2vec2's style

        Args:
            encoder: wenet's encoder,
                     only support conformer and transformer now
            embedding_dim: codebooks embedding dim
            num_embeddings: numbers of each codebook
            num_codebooks: numbers of codebooks i.e groups of codebook
            mask_prob: probs of mask
            mask_length: spans of masks
            min_maks: min masks for each audio
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
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_negatives = num_negatives

        self.features_regularization_weight = features_regularization_weight
        self.diversity_weight = diversity_weight

        # encoder
        self.encoder = encoder

        # quantizer
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

        self.mask_emb = torch.nn.parameter.Parameter(
            torch.empty(self.encoder.output_size()).uniform_(),
            requires_grad=True,
        )
        # TODO(Mddct): support causal or lookahead mask or keep consistent with
        # wenet dynamic chunk training

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
        out, _ = self._forward_encoder_blocks(masked_xs, masks, pos_emb, masks)

        gumbel_temperature = max(
            self.max_gumbel_temp * self.gumbel_temp_decay**steps,
            self.min_gumbel_temp)

        quantized_features, codevector_perplexity, _ = self.quantizer(
            unmasked_xs, masks.squeeze(1), gumbel_temperature)

        sampled_negative_indices = _sample_negative_indices(
            xs.size()[:-1], self.num_negatives, masked_masks.device,
            masked_masks)

        loss_contrastive = _compute_contrastive_loss(
            quantized_features, out, sampled_negative_indices, masked_masks,
            self.contrastive_logits_temp, self.num_negatives)
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

        return {
            "code_ppl": codevector_perplexity.detach(),
            "features_l2": features_pen,
            "loss": loss,
            "loss_contrastive": loss_contrastive / sample_size,
            "loss_diversity": loss_diversity,
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

        mask_emb = self.mask_emb.to(xs.device).view(1, 1, -1)
        xs = torch.where(masks_expand, mask_emb, xs)

        return xs, masks

    def _forward_subsampling(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def _forward_encoder_blocks(self, xs: torch.Tensor, xs_masks: torch.Tensor,
                                pos_emb: torch.Tensor, mask_pad: torch.Tensor):

        masks = xs_masks

        for layer in self.encoder.encoders:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks
