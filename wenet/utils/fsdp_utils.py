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

WENET_LAYERS_CLASSES = {
    'transformer_encoder_laer': TransformerEncoderLayer,
    'transformer_decoder_layer': DecoderLayer,
    'conformer_encoder_layer': ConformerEncoderLayer,
    'paraformer_encoder_layer': AliParaformerEncoderLayer,
    'paraformer_decoder_layer': SanmDecoderLayer,
    'squeezeformer_encoder_layer': SqueezeformerEncoderLayer,
    'ebranchformer_encoder_layer': EBranchformerEncoderLayer,
    'efficient_conformer_encoder_layer': StrideConformerEncoderLayer,
    'branchformer_encoder_layer': BranchformerEncoderLayer,

    # TODO(Mddct):
    #     1 wrap transducer's predictor and joint
    #     2 wrap paraformer's cif and ignore lstm
}


def wenet_fsdp_wrap_policy(mode):
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
            to_wrap_class.update(set(WENET_LAYERS_CLASSES.values()))
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
