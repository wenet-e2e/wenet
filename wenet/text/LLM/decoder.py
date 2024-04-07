import torch

import torch.utils.checkpoint as ckpt

from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_MLP_CLASSES)
from wenet.utils.common import mask_to_bias


class DecoderOnly(torch.nn.Module):

    def __init__(
        self,
        n_kv_head: int,
        head_dim: int,
        hidden_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        query_bias: bool = True,
        key_bias: bool = True,
        value_bias: bool = True,
        mlp_bias: bool = True,
        activation_type: str = "gelu",
        gradient_checkpointing: bool = False,
        mlp_type: str = '',
        layer_norm_type: str = 'rms_norm',
        norm_eps: float = 1e-5,
        selfattention_layer_type: str = "selfattn",
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()

        assert selfattention_layer_type in ['selfattn', 'rope_abs_selfattn']
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.decoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size,
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    attention_heads, hidden_size, attention_dropout_rate,
                    query_bias, key_bias, value_bias, use_sdpa, n_kv_head,
                    head_dim),
                mlp_class(hidden_size, linear_units, dropout_rate, activation,
                          mlp_bias),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
            ) for _ in range(num_blocks)
        ])
        self._hidden_size = hidden_size
        self.gradient_checkpoint = gradient_checkpointing
        self.use_sdpa = use_sdpa

    def forward(self, input: torch.Tensor, att_mask: torch.Tensor):
        xs = self.embed(input)
        xs, pos_emb = self.pos_enc(xs)
        tgt_mask = att_mask
        if self.use_sdpa:
            tgt_mask = mask_to_bias(tgt_mask, xs.dtype)
        if not self.gradient_checkpoint:
            decoder_out, _, _, _ = self.decoders(xs,
                                                 tgt_mask,
                                                 pos_emb,
                                                 mask_pad=None)
        else:
            assert self.training
            decoder_out = xs
            for layer in self.decoders:
                decoder_out, _, _, _ = ckpt.checkpoint(
                    layer.__call__,
                    decoder_out,
                    tgt_mask,
                    pos_emb,
                )

        return decoder_out

    @property
    def hidden_size(self):
        return self._hidden_size
