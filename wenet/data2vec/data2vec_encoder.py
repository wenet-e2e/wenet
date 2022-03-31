#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# The origial Data2vec work is in:
# Paper: https://arxiv.org/pdf/2202.03555.pdf
# Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/data2vec

"""Encoder definition."""
from typing import Tuple, List, Optional
import logging
import torch
import math
from typeguard import check_argument_types
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask
from wenet.utils.mask import compute_mask_indices
from wenet.data2vec.ema import EMA

def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining

class Data2vecBaseEncoder(torch.nn.Module):
    def __init__(
        self,
        data2vec_conf:dict,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        ema: EMA = None
    ):

        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)
        self.feature_norm = torch.nn.LayerNorm(output_size, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk

        self.embed_dim=output_size
        final_dim=output_size
        self.encoder_embed_dim=output_size
        
        self.mask_prob = data2vec_conf.get('mask_prob', 0.65)
        self.mask_selection = "static"
        self.mask_other = 0
        self.mask_length = 10
        self.no_mask_overlap =False
        self.mask_min_space = 1

        self.mask_channel_prob = data2vec_conf.get('mask_channel_prob', 0.0)
        self.mask_channel_selection = "static"
        self.mask_channel_other = 0
        self.mask_channel_length =  data2vec_conf.get('mask_channel_length', 10)
        self.no_mask_channel_overlap = False
        self.mask_channel_min_space = 1

        self.ema = None
        self.embed_dim=self.encoder_embed_dim
        self.ema_decay=data2vec_conf.get("ema_decay",0.999)
        self.ema_end_decay=data2vec_conf.get("ema_end_decay",0.9999)
        self.ema_transformer_only=data2vec_conf.get("ema_transformer_only",True)
        self.ema_layers_only=data2vec_conf.get("ema_layers_only",True)
        self.ema_anneal_end_step=data2vec_conf.get("ema_anneal_end_step",30000)
        
        self.min_target_var=0.1
        self.min_pred_var=0.01

        self.layer_norm_target_layer: bool = False
        self.instance_norm_target_layer: bool = True
        self.instance_norm_targets: bool = False
        self.layer_norm_targets: bool = False
        self.batch_norm_target_layer: bool = False
        self.group_norm_target_layer: bool = False

        self.average_top_k_layers = data2vec_conf.get("average_top_k_layers",8)
        self.loss_beta = data2vec_conf.get("loss_beta",0)
        self.loss_scale = data2vec_conf.get("loss_scale",None)
        
        self.project_final=data2vec_conf.get('project_final', False)
        self.intermediate_layers=data2vec_conf.get('intermediate_layers',None)

        self.mask=data2vec_conf.get('mask', True)

        self.pretrain=data2vec_conf.get('pretrain',True)

        self.final_proj = torch.nn.Linear(output_size, final_dim)

        self.target_glu=None
        self.logit_temp=0.1

        self.mask_emb = torch.nn.Parameter(
            torch.FloatTensor(self.encoder_embed_dim).uniform_()
        )
        self.feature_grad_mult=data2vec_conf.get('feature_grad_mult',1.0)

        self.num_updates=0


    def output_size(self) -> int:
        return self._output_size

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def make_ema_teacher(self):
        ema_config = {
            "ema_decay": self.ema_decay,
            "ema_fp32" :True,
            "store_ema" : False,
            "ema_start_update" : 0 ,
            "ema_update_freq" : 1,
        }
        skip_keys = set()
        if self.ema_layers_only:
            self.ema_transformer_only = True
            skip_keys.add("embed.")

        self.ema = EMA(
            self.encoders,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):

        if self.ema is None and self.final_proj is not None:
            logging.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None and num_updates!= self.num_updates :
            if self.ema_decay != self.ema_end_decay:
                if num_updates >= self.ema_anneal_end_step:
                    decay = self.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.ema_decay,
                        self.ema_end_decay,
                        num_updates,
                        self.ema_anneal_end_step,
                    )
                self.ema._set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoders)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        logging.info(f"state_dict")
        # if self.ema is not None:
        #     state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        logging.info(f"_load_from_state_dict")
        if self.ema is not None:
            logging.info(f"_load_from_state_dict ema!")
            k = prefix + "_ema"
            assert k in state_dict
            logging.info(k)
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def extract_features(
        self,
        x: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
    ):
        layer_idx=0
        inter_idx=0
        intermediate_outputs = []
        for layer in self.ema.model:
            x, chunk_masks,_,layer_out = layer(x, chunk_masks, pos_emb)
            encoder_output = x
            if (
                    self.intermediate_layers is not None
                    and inter_idx <len(self.intermediate_layers)
                    and layer_idx  == int(self.intermediate_layers[inter_idx])
                    
                ):
                if self.normalize_before:
                    encoder_output = self.after_norm(encoder_output)
                intermediate_outputs.append((encoder_output,layer_out))
                inter_idx=inter_idx+1
            layer_idx=layer_idx+1

        if self.normalize_before:
            x = self.after_norm(x)

        return x,intermediate_outputs

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        features_only: bool=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        if self.feature_grad_mult > 0:
            features, pos_emb, masks = self.embed(xs, masks)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features, pos_emb, masks = self.embed(xs, masks)
        
        chunk_masks=masks
        #L2 loss pen 
        features_pen = features.float().pow(2).mean()

        features = self.feature_norm(features)
        unmasked_features = features.clone()
        input_features=None

        if self.mask and self.pretrain:

            x, mask_indices = self.apply_mask(features, None)
            input_features=x.clone()
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        elif self.mask and self.training:
            x, mask_indices = self.apply_mask(features, None)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
        else:
            y = unmasked_features
            x = features
            y = unmasked_features
            mask_indices = None

        for layer in self.encoders:
            x, chunk_masks, _ , _ = layer(x, chunk_masks, pos_emb)
           
        if self.normalize_before:
            x = self.after_norm(x)
        ext_result={"outputs":x}

        if features_only:
            return x, masks,ext_result

        with torch.no_grad():
            self.ema.model.eval()

            if self.ema_transformer_only:
                y, layer_results = self.extract_features(
                    unmasked_features,
                    chunk_masks,
                    pos_emb,
                )
                y = {
                    "x": y,
                    "layer_results": layer_results,
                }

            target_layer_results = [l[1] for l in y["layer_results"]]

            permuted = False
            if self.instance_norm_target_layer or self.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(0, 2, 1) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            # if not permuted:
            #     y = y.transpose(0, 1)

            y = y[mask_indices]

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)
        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)

        ext_result["losses_reg"] = loss.sum() * scale

        if "sample_size" not in ext_result:
            ext_result["sample_size"] = loss.numel()
        
        #bug: uio mode trainging will be hold on

        # with torch.no_grad():
        #     ext_result["target_var"] = self.compute_var(y)
        #     ext_result["pred_var"] = self.compute_var(x.float())

        # if self.num_updates > 5000 and ext_result["target_var"] < self.min_target_var:
        #     logging.warning(
        #         f"target var is {ext_result['target_var'].item()} < {self.min_target_var}, exiting"
        #     )
        #     # raise Exception(
        #     #     f"target var is {ext_result['target_var'].item()} < {self.min_target_var}, exiting"
        #     # )
        # if self.num_updates > 5000 and ext_result["pred_var"] < self.min_pred_var:
        #     logging.warning(
        #         f"pred var is {ext_result['pred_var'].item()} < {self.min_pred_var}, exiting"
        #     )
        #     # raise Exception(
        #     #     f"pred var is {ext_result['pred_var'].item()} < {self.min_pred_var}, exiting"
        #     # )

        # if self.ema is not None:
        #     ext_result["ema_decay"] = self.ema.get_decay() * 1000

        return x,masks,ext_result
    
    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()
    
    def forward_mask(
        self,
        xs: torch.Tensor,
        masks: torch.Tensor,
        decoding_chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    # def forward(
    #     self,
    #     xs: torch.Tensor,
    #     xs_lens: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk, it's
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
        Returns:
            encoder output tensor, lens and mask
        """
        # batch, max_len = torch.tensor(xs.shape[:2]).tolist()
    

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)

        chunk_masks=masks

        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb)
        if self.normalize_before:
            xs = self.after_norm(xs)
   
        return xs, masks

  
class Data2vecConformerEncoder(Data2vecBaseEncoder):
    """Conformer encoder module."""
    def __init__(
        self,
        data2vec_conf:dict,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__(data2vec_conf,input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn)
        activation = get_activation(activation_type)

        # self-attention module definition
        if selfattention_layer_type == "rel_selfattn":
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        else:
            encoder_selfattn_layer = MultiHeadedAttention

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                ffn_res=True,
            ) for _ in range(num_blocks)
        ])
