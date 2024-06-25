from functools import partial
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint as ckpt
from wenet.transformer.attention import T_CACHE

from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_EMB_CLASSES, WENET_MLP_CLASSES,
                                     WENET_NORM_CLASSES)
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
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        query_bias: bool = False,
        key_bias: bool = False,
        value_bias: bool = False,
        mlp_bias: bool = False,
        activation_type: str = "gelu",
        gelu_approximate: Union[str, None] = None,
        max_position_embeding: int = 8192,
        mlp_type: str = 'gated',
        layer_norm_type: str = 'rms_norm',
        norm_eps: float = 1e-5,
        rms_norm_offset: bool = True,
        selfattention_layer_type: str = "rope_abs_selfattn",
        use_sdpa: bool = False,
        gradient_checkpointing: bool = False,
        rope_theta: float = 10000.0,
        rope_style: str = 'google',
        scale_embed: bool = True,
    ) -> None:
        super().__init__()

        assert selfattention_layer_type in ['rope_abs_selfattn']
        self.pos_enc = WENET_EMB_CLASSES["rope_pos"](
            hidden_size,
            head_dim,
            max_len=max_position_embeding,
            dropout_rate=positional_dropout_rate,
            rope_theta=rope_theta,
            scale=scale_embed)
        if activation_type == "gelu" and gelu_approximate is not None:
            activation = WENET_ACTIVATION_CLASSES['gelu'](
                approximate=gelu_approximate)
        else:
            activation = WENET_ACTIVATION_CLASSES[activation_type]()

        mlp_class = WENET_MLP_CLASSES[mlp_type]
        self.num_blocks = num_blocks
        # TODO: support lora & refactor lora
        self.decoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size,
                WENET_ATTENTION_CLASSES[selfattention_layer_type](
                    attention_heads,
                    hidden_size,
                    attention_dropout_rate,
                    query_bias,
                    key_bias,
                    value_bias,
                    use_sdpa,
                    n_kv_head,
                    head_dim,
                    style=rope_style),
                mlp_class(hidden_size, linear_units, dropout_rate, activation,
                          mlp_bias),
                dropout_rate,
                normalize_before,
                layer_norm_type=layer_norm_type,
                norm_eps=norm_eps,
                rms_norm_offset=rms_norm_offset,
            ) for _ in range(self.num_blocks)
        ])
        self.pre_norm = normalize_before
        self.final_norm: Optional[torch.nn.Module] = None
        if self.pre_norm:
            norm_class = WENET_NORM_CLASSES[layer_norm_type]
            if layer_norm_type == "rms_norm":
                norm_class = partial(
                    norm_class,
                    add_unit_offset=rms_norm_offset,
                )
            self.final_norm = norm_class(hidden_size, eps=norm_eps)

        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self._hidden_size = hidden_size
        self.use_sdpa = use_sdpa
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        input: torch.Tensor,
        att_mask: torch.Tensor,
        input_position: Union[int, torch.Tensor] = 0,
        kv_caches: Optional[List[T_CACHE]] = None,
    ) -> Tuple[torch.Tensor, Union[List[T_CACHE], None]]:
        xs, pos_emb = self.pos_enc(input, offset=input_position)
        if self.use_sdpa:
            att_mask = mask_to_bias(att_mask, xs.dtype)

        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, att_mask, pos_emb)
        else:
            xs, kv_caches = self.forward_layers(xs, att_mask, pos_emb,
                                                kv_caches)
        if self.pre_norm and self.final_norm is not None:
            xs = self.final_norm(xs)
        return xs, kv_caches

    def forward_layers(
        self,
        xs: torch.Tensor,
        att_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        kv_caches: Optional[List[T_CACHE]] = None,
    ) -> Tuple[torch.Tensor, Union[List[T_CACHE], None]]:
        if self.training:
            for (i, layer) in enumerate(self.decoders):
                xs, _, _, _ = layer(xs, att_mask, pos_emb)
            new_kv_caches = kv_caches
        else:
            assert kv_caches is not None
            new_kv_caches = []
            for (i, layer) in enumerate(self.decoders):
                xs, _, new_kv_cache, _ = layer(xs,
                                               att_mask,
                                               pos_emb,
                                               att_cache=(kv_caches[i][0],
                                                          kv_caches[i][1]))
                new_kv_caches.append(new_kv_cache)

        return xs, new_kv_caches

    @torch.jit.ignore(drop=True)
    def forward_layers_checkpointed(self, xs: torch.Tensor,
                                    att_mask: torch.Tensor,
                                    pos_emb: torch.Tensor) -> torch.Tensor:
        for layer in self.decoders:
            xs, _, _, _ = ckpt.checkpoint(layer.__call__, xs, att_mask,
                                          pos_emb)
        return xs

    @property
    def hidden_size(self):
        return self._hidden_size
