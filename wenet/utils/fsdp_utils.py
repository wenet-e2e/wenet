from functools import partial
import os
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                    FullStateDictConfig, StateDictType)

from torch.distributed.fsdp.wrap import (_or_policy, lambda_auto_wrap_policy,
                                         transformer_auto_wrap_policy)
from wenet.utils.checkpoint import save_state_dict_and_infos

from wenet.utils.init_model import (WENET_DECODER_CLASSES,
                                    WENET_ENCODER_CLASSES)


def wenet_fsdp_wrap_policy():
    to_wrap_class = set(WENET_ENCODER_CLASSES.values())
    to_wrap_class.update(set(WENET_DECODER_CLASSES.values()))
    # TODO(Mddct):
    #     1 wrap transducer's predictor and joint
    #     2 wrap paraformer's cif

    wrap_policy = partial(transformer_auto_wrap_policy,
                          transformer_layer_cls=to_wrap_class)

    # https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/fsdp_utils.py#L13 # noqa
    def no_grad_fn(module):
        if (len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad):
            return True
        return False

    no_grad_ploicy = partial(lambda_auto_wrap_policy, lambda_fn=no_grad_fn)

    auto_wrap_policy = partial(_or_policy,
                               policies=[no_grad_ploicy, wrap_policy])
    return auto_wrap_policy


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
