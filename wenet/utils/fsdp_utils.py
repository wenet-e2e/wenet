from functools import partial
import os
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                    FullStateDictConfig, StateDictType)

from torch.distributed.fsdp.wrap import (lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from wenet.branchformer.encoder_layer import BranchformerEncoderLayer
from wenet.e_branchformer.encoder_layer import EBranchformerEncoderLayer
from wenet.efficient_conformer.encoder_layer import StrideConformerEncoderLayer
from wenet.paraformer.layers import AliParaformerEncoderLayer, SanmDecoderLayer
from wenet.squeezeformer.encoder_layer import SqueezeformerEncoderLayer
from wenet.transformer.encoder_layer import (ConformerEncoderLayer,
                                             TransformerEncoderLayer)
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.utils.checkpoint import save_state_dict_and_infos
from wenet.utils.init_model import WENET_DECODER_CLASSES, WENET_ENCODER_CLASSES

WENET_ENCODER_LAYERS_CLASSES = {
    'transformer_encoder_layer': TransformerEncoderLayer,
    'conformer_encoder_layer': ConformerEncoderLayer,
    'paraformer_encoder_layer': AliParaformerEncoderLayer,
    'squeezeformer_encoder_layer': SqueezeformerEncoderLayer,
    'ebranchformer_encoder_layer': EBranchformerEncoderLayer,
    'efficient_conformer_encoder_layer': StrideConformerEncoderLayer,
    'branchformer_encoder_layer': BranchformerEncoderLayer,
}

WENET_DECODER_LAYERS_CLASSES = {
    'transformer_decoder_layer': DecoderLayer,
    'paraformer_decoder_layer': SanmDecoderLayer,
    # TODO(Mddct):
    #     1 wrap transducer's predictor and joint
    #     2 wrap paraformer's cif and ignore lstm
}


def wenet_fsdp_wrap_policy(mode):
    # different wrap methods
    # please refer： https://openmmlab.medium.com/its-2023-is-pytorch-s-fsdp-the-best-choice-for-training-large-models-fe8d2848832f # noqa
    assert mode in ['no_shard', 'model', 'zero2', 'zero3']
    if mode == 'no_shard':
        return None
    else:
        # TODO(Mddct):  Support user customization
        # see more wrap methods:
        # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/fsdp_utils.py#L13 # noqa
        if mode == 'model':
            enc_dec_wrap_policy = partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda module: isinstance(
                    module,
                    tuple(WENET_ENCODER_CLASSES.values()) + tuple(
                        WENET_DECODER_CLASSES.values())))
            return enc_dec_wrap_policy
        else:
            to_wrap_class = set()
            to_wrap_class.update(set(WENET_ENCODER_LAYERS_CLASSES.values()))
            to_wrap_class.update(set(WENET_DECODER_LAYERS_CLASSES.values()))
            layers_wrap_policy = partial(transformer_auto_wrap_policy,
                                         transformer_layer_cls=to_wrap_class)
            return layers_wrap_policy


fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True,
                                            rank0_only=True)


def fsdp_save_model(model, save_model_path, info_dict):
    # TODO(Mddct); When the model is large, saving a model will take a long time.
    # We only need to keep the sharding in an asynchronous manner, but it is
    # good now. This feature will be supported when llm is supported in the future.

    rank = int(os.environ.get('RANK', 0))
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                              fullstate_save_policy):
        state_dict = model.state_dict()
        if rank == 0:
            save_state_dict_and_infos(state_dict, save_model_path, info_dict)


def check_gradient_checkpoint(model):
    ckpt_laye_types = []
    if hasattr(model, 'encoder') and hasattr(model.encoder,
                                             'gradient_checkpointing'):
        if model.encoder.gradient_checkpointing:
            model.encoder.gradient_checkpointing = False
            ckpt_laye_types += list(WENET_ENCODER_LAYERS_CLASSES.values())
    if hasattr(model, 'decoder') and hasattr(model.decoder,
                                             'gradient_checkpointing'):
        if model.decoder.gradient_checkpointing:
            model.decoder.gradient_checkpointing = False
            ckpt_laye_types += list(WENET_DECODER_LAYERS_CLASSES.values())
    return tuple(ckpt_laye_types)


def apply_fsdp_checkpointing(model, ckpt_layer_types: tuple):
    # NOTE(Mddct):  torch.utils.checkpoint is currently incompatible with
    # wenet's model mode. Using this writing method, Please refer to
    # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/policies/activation_checkpointing_functions.py#L21 # noqa
    if len(ckpt_layer_types) == 0:
        return
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, ckpt_layer_types))
