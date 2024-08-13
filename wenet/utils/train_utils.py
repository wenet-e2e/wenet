# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
import copy
from typing import List, Optional

import deepspeed
import json
import logging
import os
import torch
import yaml

import torch.optim as optim
import torch.distributed as dist

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                    CPUOffload, MixedPrecision,
                                    sharded_grad_scaler, ShardingStrategy)
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live)
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live)
from deepspeed.utils.zero_to_fp32 import (
    convert_zero_checkpoint_to_fp32_state_dict)
from wenet.utils.checkpoint import save_checkpoint
from wenet.utils.common import (StepTimer, get_nested_attribute, lrs_to_str,
                                tensor_to_scalar)
from wenet.utils.fsdp_utils import (check_gradient_checkpoint, fsdp_save_model,
                                    apply_fsdp_checkpointing,
                                    wenet_fsdp_wrap_policy)
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.common import TORCH_NPU_AVAILABLE
from wenet.utils.init_dataset import init_dataset


def add_model_args(parser):
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument(
        '--enc_init_mods',
        default="encoder.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument(
        '--freeze_modules',
        default="",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help='free module names',
    )
    return parser


def add_trace_args(parser):
    parser.add_argument('--jit',
                        action='store_true',
                        default=False,
                        help='if use jit to trace model while training stage')
    parser.add_argument('--print_model',
                        action='store_true',
                        default=False,
                        help='print model')
    return parser


def add_dataset_args(parser):
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    return parser


def add_lora_args(parser):
    '''Configure parameters for LoRA fine-tuning. Set use_lora and
       only_optimize_lora to true to enable LoRA functionality.
       LoRA will be injected to model through (lora_modules, lora_attn_attr,
       lora_list).
       LoRA weights will be merged after calling model.eval()
       (or model.train(mode=False)).
       LoRA weights need to be loaded after fine-tuning with DeepSpeed.
    '''
    parser.add_argument("--use_lora",
                        default=False,
                        type=bool,
                        help="whether use the lora finetune.")
    parser.add_argument("--only_optimize_lora",
                        default=False,
                        type=bool,
                        help="freeze all other paramters and only optimize \
                        LoRA-related prameters.")
    parser.add_argument(
        '--lora_modules',
        default="encoder.encoders",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help='modules names needs inject lora',
    )
    parser.add_argument(
        "--lora_attn_attr",
        default="self_attn,src_attn",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="lora_attn_attr.")
    parser.add_argument(
        "--lora_list",
        default="linear_out,linear_q,linear_k,linear_v",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="lora module list.")
    parser.add_argument("--lora_rank",
                        default=8,
                        type=int,
                        help="lora rank num.")
    parser.add_argument("--lora_alpha",
                        default=8,
                        type=int,
                        help="lora scale param, scale=lora_alpha/lora_rank.")
    parser.add_argument("--lora_dropout",
                        default=0,
                        type=float,
                        help="lora dropout param.")
    parser.add_argument("--lora_ckpt_path",
                        default=None,
                        type=str,
                        help="lora checkpoint path.")
    parser.add_argument("--lora_reinit",
                        default=False,
                        type=bool,
                        help="whether use the lora init, default is zero init.")
    parser.add_argument('--lora_init_yaml',
                        default="wenet/finetune/lora/config.yaml",
                        type=str,
                        help='Path to the configuration YAML file')
    return parser


def add_ddp_args(parser):
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo', "hccl"],
                        help='distributed backend')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    return parser


def add_deepspeed_args(parser):
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of wenet_join. ' +
                        '30s for aishell & 300s for wenetspeech')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser = deepspeed.add_config_arguments(parser)
    return parser


def add_fsdp_args(parser):
    parser.add_argument(
        '--dtype',
        default='fp32',
        choices=['fp32', 'fp16', 'bf16'],
        help='when amp is used, dtype is automatically set to fp16.\
        this arg has no effect when deepspeed is enabled.')
    parser.add_argument(
        '--fsdp_cpu_offload',
        default=False,
        type=bool,
        help='whether to offload parameters to CPU',
    )
    parser.add_argument(
        '--fsdp_sync_module_states',
        type=bool,
        default=True,
        help='\
        each FSDP module will broadcast module parameters and buffers from \
        rank 0 to ensure that they are replicated across ranks',
    )
    parser.add_argument(
        '--fsdp_sharding_strategy',
        default='zero2',
        # TODO(Mddct): pipeline and model parallel (3-D parallelism)
        choices=['no_shard', 'model', 'zero2', 'zero3'],
        help='Sharding strategy for FSDP. Choose from the following options:\n'
        '  - "no_shard": Equivalent to DistributedDataParallel (DDP).\n'
        '  - "model": WENET_ENC_DEC strategy, equivalent to DeepSpeed zero1.\n'
        '  - "zero2": SHARD_GRAD_OP strategy, equivalent to DeepSpeed zero2.\n'
        '  - "zero3": FULL_SHARD strategy, equivalent to DeepSpeed zero3.\n'
        'For more information, refer to the FSDP API documentation.')
    return parser


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if args.train_engine in ["torch_ddp", "torch_fsdp"]:
        if "cuda" in args.device:
            torch.cuda.set_device(local_rank)
        elif "npu" in args.device and TORCH_NPU_AVAILABLE:
            torch.npu.set_device(local_rank)
        else:
            logging.error("not supported device: {}".format(args.device))
        dist.init_process_group(args.dist_backend)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend)
    else:
        logging.error("not supported engine: {}".format(args.train_engine))
    return world_size, local_rank, rank


def check_modify_and_save_config(args, configs, symbol_table):
    if args.train_engine in ["torch_ddp", "torch_fsdp"]:
        if args.use_amp:
            configs["dtype"] = "fp16"
            args.dtype = 'fp16'
        else:
            configs["dtype"] = args.dtype
    elif args.train_engine == "deepspeed":
        # NOTE(xcsong): DeepSpeed does not support uneven data. When using custom
        #   dataset, we need to manually ensure that the data is evenly distributed
        #   across all processe. we impl `train_utils.py::wenet_join` for this func
        #   ref: https://github.com/microsoft/DeepSpeed/issues/2223
        #
        # NOTE(xsong):  We also need to keep:
        #       1. `train_micro_batch_size_per_gpu == 1`
        #       2. `accum_grad (in train_confomrer.yaml)
        #               == gradient_accumulation_steps (in ds_config.json)`
        #       3. `grad_clip (in train_confomrer.yaml)
        #               == gradient_clipping (in ds_config.json)`
        #   The reason for such consistence checking lies in that deepspeed's native
        #   dataloader uses PyTorch's torch.utils.data.DistributedSampler which does
        #   not support IterableDataset, IterableDataset is extremly useful in large
        #   scale training because it lets you stream the data without having to
        #   download the complete dataset.
        #       ref: https://github.com/microsoft/DeepSpeed/issues/1371
        #           https://github.com/microsoft/DeepSpeed/issues/285
        #   To make deepspeed training compatible with IterableDataset, we have to
        #   use custom dataloader instead of deepspeed's native loader and thus we
        #   should configure batchsize in train_confomrer.yaml instead of
        #   ds_config.json. On the contrary, gradient accumulation / clipping should be
        #   configured in ds_config.json since they will be handled by ds automatically.
        #       ref: https://github.com/microsoft/DeepSpeed/issues/62
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs["dtype"] = "bf16"
        else:
            configs["dtype"] = "fp32"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        assert ds_configs["gradient_accumulation_steps"] == configs[
            'accum_grad']
        assert ds_configs["gradient_clipping"] == configs['grad_clip']
        assert ds_configs["steps_per_print"] == configs['log_interval']

    if args.use_lora:
        configs['lora_conf'] = {}
        configs['lora_conf']['lora_modules'] = args.lora_modules
        configs['lora_conf']['lora_attn_attr'] = args.lora_attn_attr
        configs['lora_conf']['lora_list'] = args.lora_list
        configs['lora_conf']['lora_rank'] = args.lora_rank
        configs['lora_conf']['lora_alpha'] = args.lora_alpha
        configs['lora_conf']['lora_dropout'] = args.lora_dropout

    if configs["model"] == 'asr_model':
        if 'input_dim' not in configs:
            if 'fbank_conf' in configs['dataset_conf']:
                input_dim = configs['dataset_conf']['fbank_conf'][
                    'num_mel_bins']
            elif 'log_mel_spectrogram_conf' in configs['dataset_conf']:
                input_dim = configs['dataset_conf'][
                    'log_mel_spectrogram_conf']['num_mel_bins']
            else:
                input_dim = configs['dataset_conf']['mfcc_conf'][
                    'num_mel_bins']
        else:
            input_dim = configs['input_dim']

        configs['input_dim'] = input_dim

    configs, _ = get_blank_id(configs, symbol_table)
    configs['output_dim'] = configs['vocab_size']

    configs['train_engine'] = args.train_engine
    configs['use_amp'] = args.use_amp
    configs['model_dir'] = args.model_dir
    configs['save_states'] = args.save_states

    # Save configs to model_dir/train.yaml for inference and export
    if int(os.environ.get('RANK', 0)) == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    if configs["model_conf"].get("apply_non_blank_embedding", False):
        logging.warn('Had better load a well trained model'
                     'if apply_non_blank_embedding is true !!!')

    return configs


def init_dataset_and_dataloader(args, configs, tokenizer, seed=777):
    generator = torch.Generator()
    generator.manual_seed(seed)

    # if save_interval in configs, steps mode else epoch mode
    if "save_interval" in configs:
        configs['dataset_conf']['cycle'] = configs.get('max_epoch', 100)
    conf = configs['dataset_conf']
    dataset_type = configs.get('dataset', 'asr')
    configs['vocab_size'] = tokenizer.vocab_size()
    train_dataset = init_dataset(dataset_type,
                                 args.data_type,
                                 args.train_data,
                                 tokenizer,
                                 conf,
                                 True,
                                 split='train')
    cv_dataset = init_dataset(dataset_type,
                              args.data_type,
                              args.cv_data,
                              tokenizer,
                              conf,
                              partition=False,
                              split='cv')

    # NOTE(xcsong): Why we prefer persistent_workers=True ?
    #   https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   persistent_workers=True,
                                   generator=generator,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                persistent_workers=True,
                                generator=generator,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def wrap_cuda_model(args, model, configs=None):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if hasattr(model, 'encoder'):
        grad_ckpt = getattr(model.encoder, 'gradient_checkpointing', False)
    else:
        grad_ckpt = False
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        device = torch.device(args.device)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=not grad_ckpt)
    elif args.train_engine == "deepspeed":  # deepspeed
        # NOTE(xcsong): look in detail how the memory estimator API works:
        #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model,
                num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
        device = torch.device(args.device)  # Init device later
        pass  # Init DeepSpeed later
    elif args.train_engine == 'torch_fsdp':
        assert configs is not None
        mixed_precision_dtype = {
            'fp32': torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[configs['dtype']]

        sharding_strategy = {
            'model': ShardingStrategy.SHARD_GRAD_OP,
            'zero2': ShardingStrategy.SHARD_GRAD_OP,
            'zero3': ShardingStrategy.FULL_SHARD,
            'no_shard': ShardingStrategy.NO_SHARD,
        }[args.fsdp_sharding_strategy]
        wrap_policy = wenet_fsdp_wrap_policy(mode=args.fsdp_sharding_strategy)
        layer_types = check_gradient_checkpoint(model)
        if "cuda" in args.device:
            device_id = torch.cuda.current_device()
        elif "npu" in args.device and TORCH_NPU_AVAILABLE:
            device_id = torch.npu.current_device()
        else:
            logging.error("not supported device: {}".format(args.device))
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            cpu_offload=CPUOffload(offload_params=True)
            if args.fsdp_cpu_offload is True else None,
            mixed_precision=MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=mixed_precision_dtype,
                buffer_dtype=mixed_precision_dtype,
            ),
            sharding_strategy=sharding_strategy,
            limit_all_gathers=True,
            use_orig_params=True,
            sync_module_states=args.fsdp_sync_module_states,
            # init_distributed is called (torch.cuda.set_device),
            # we should set device_id, see FSDP api
            device_id=device_id)
        apply_fsdp_checkpointing(model, layer_types)
        device = torch.device(args.device)
    else:
        logging.error("not supported engine: {}".format(args.train_engine))
    if args.train_engine in ["torch_fsdp", "torch_ddp"]:
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks, )
            model.register_comm_hook(state=None,
                                     hook=comm_hooks.fp16_compress_hook)

    return model, device


def init_optimizer_and_scheduler(args, configs, model):
    groups = []
    lr = configs['optim_conf'].get('lr')
    if isinstance(lr, List):
        assert configs['scheduler'] == 'warmuplr'
        modules_m = configs['optim_conf']['modules']
        assert isinstance(modules_m, List)
        assert len(modules_m) + 1 == len(lr)
        special_param_ids = set()
        rest_params = []
        for (i, m_str) in enumerate(modules_m):
            sub_module = get_nested_attribute(model, m_str)
            subs_params = []
            for _, sub_params in sub_module.named_parameters():
                subs_params.append(sub_params)
                special_param_ids.add(id(sub_params))
            groups.append({'params': subs_params, 'lr': lr[i]})
        # other model's parameters
        for _, param in model.named_parameters():
            if id(param) not in special_param_ids:
                rest_params.append(param)
        groups.append({'params': rest_params, 'lr': lr[-1]})

    params = groups if len(groups) > 0 else model.parameters()
    optim_conf = copy.deepcopy(configs['optim_conf'])
    if 'modules' in optim_conf:
        del optim_conf['modules']
    if isinstance(lr, List):
        optim_conf['lr'] = lr[-1]
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(params, **optim_conf)
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(params, **optim_conf)
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])

    scheduler_type = None
    if configs['scheduler'] == 'warmuplr':
        scheduler_type = WarmupLR
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    elif configs['scheduler'] == 'NoamHoldAnnealing':
        scheduler_type = NoamHoldAnnealing
        scheduler = NoamHoldAnnealing(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    # NOTE(xcsong): Custom optimizer might yield poor performance when
    #   zero-offload is enabled, if you do want to offload optimizer to CPU,
    #   please set optimizer in ds_config.json, see:
    #   (https://www.deepspeed.ai/docs/config-json/#optimizer-parameters)
    if args.train_engine == "deepspeed":
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "optimizer" in ds_configs:
            # NOTE(xcsong): Disable custom optimizer if it is set in ds_config,
            # extremely useful when enable cpu_offload, DeepspeedCpuAdam
            # could be 4~5x faster than torch native adam
            optimizer = None
            if "scheduler" in ds_configs:
                scheduler = None
            else:

                def scheduler(opt):
                    return scheduler_type(opt, **configs['scheduler_conf'])

        model, optimizer, _, scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            model_parameters=model.parameters())

    step = configs["init_infos"].get("step", -1)
    scheduler.set_step(step)
    return model, optimizer, scheduler


def trace_and_print_model(args, model):
    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if int(os.environ.get('RANK', 0)) == 0:
        if args.jit:
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(args.model_dir, 'init.zip'))
        if args.print_model:
            print(model)
            num_params = sum(p.numel() for p in model.parameters())
            print('the number of model params: {:,d}'.format(num_params))


def init_summarywriter(args):
    writer = None
    if int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(args.model_dir, exist_ok=True)
        exp_id = os.path.basename(args.model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))
    return writer


def init_scaler(args):
    scaler = None
    if args.use_amp:
        if "cuda" in args.device:
            scaler = torch.cuda.amp.GradScaler()
        elif "npu" in args.device and TORCH_NPU_AVAILABLE:
            scaler = torch.npu.amp.GradScaler()
        else:
            logging.error("not supported device: {}".format(args.device))
    elif args.train_engine == 'torch_fsdp':
        # why bf16 don't need scaler:
        # https://discuss.pytorch.org/t/why-bf16-do-not-need-loss-scaling/176596
        if args.dtype in ['fp16']:
            scaler = sharded_grad_scaler.ShardedGradScaler(enabled=True)
    return scaler


def save_model(model, info_dict):
    rank = int(os.environ.get('RANK', 0))
    tag = info_dict["tag"]
    model_dir = info_dict["model_dir"]
    save_model_path = os.path.join(model_dir, '{}.pt'.format(tag))
    # save ckpt
    if info_dict["train_engine"] == "deepspeed":
        # NOTE(xcsong): All ranks should call this API, but only rank 0
        #   save the general model params. see:
        #   https://github.com/microsoft/DeepSpeed/issues/2993
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir,
                                  tag=tag,
                                  client_state=info_dict)
            if info_dict["save_states"] == "model_only" and rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(model_dir,
                                                           save_model_path,
                                                           tag=tag)
                os.system("rm -rf {}/{}".format(model_dir, tag))

    elif info_dict['train_engine'] == "torch_fsdp":
        fsdp_save_model(model, save_model_path, info_dict)
    elif rank == 0:
        # NOTE(xcsong): For torch_ddp, only rank-0 should call this.
        save_checkpoint(model, save_model_path, info_dict)
    # save yaml
    if rank == 0:
        with open("{}/{}.yaml".format(model_dir, tag), 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)


def wenet_join(group_join, info_dict):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    train_engine = info_dict.get('train_engine', "torch_ddp")

    if info_dict["batch_idx"] == 0 or train_engine == "torch_ddp":
        # NOTE(xcsong): skip first batch because its processing time includes
        #   dataloader initialization time, which may exceed 30 seconds
        return False

    try:
        # NOTE(xcsong): Why we need a new group?
        #   Because Deepspeed has its own group where all the relevant communication
        #   operations are executed. If we add a communication operation that is not
        #   managed by Deepspeed in this group, it's highly likely to cause
        #   communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join,
                               timeout=group_join.options._timeout)
    except RuntimeError as e:
        logging.info("Detected uneven workload distribution: {}\n".format(e) +
                     "Break current worker to manually join all workers, " +
                     "world_size {}, current rank {}, current local_rank {}\n".
                     format(world_size, rank, local_rank))
        return True

    return False


def batch_forward(model, batch, scaler, info_dict, device):
    train_engine = info_dict.get('train_engine', "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1)

    dtype = info_dict.get("dtype", "fp32")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = None

    # autocast context
    # The more details about amp can be found in
    # https://pytorch.org/docs/stable/notes/amp_examples.html
    amp_autocast = torch.cuda.amp.autocast
    if "npu" in device.__str__() and TORCH_NPU_AVAILABLE:
        amp_autocast = torch.npu.amp.autocast
    autocast = {
        "deepspeed":
        amp_autocast(enabled=dtype is not None,
                     dtype=dtype,
                     cache_enabled=False),
        "torch_ddp":
        amp_autocast(enabled=scaler is not None),
        "torch_fsdp":
        amp_autocast(enabled=True, dtype=dtype)
        if dtype is not None else nullcontext()
    }[train_engine]
    with autocast:
        loss_dict = model(batch, device)

    info_dict['loss_dict'] = loss_dict
    return info_dict


def batch_backward(model, scaler, info_dict):
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1)
    use_amp = info_dict.get('use_amp', False)
    if use_amp:
        assert scaler is not None
    loss = info_dict['loss_dict']['loss']

    if train_engine == "deepspeed":
        # NOTE(xcsong): `model.backward(loss)` is equivalent to
        #               `scale_loss_wrt_accum_grad + loss.backward()`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        scaled_loss = model.backward(loss)
    else:
        assert train_engine in ["torch_ddp", "torch_fsdp"]
        scaled_loss = loss / accum_grad
        if scaler is not None:
            # fp16 (amp and fsdp)
            scaler.scale(scaled_loss).backward()
        else:
            # float32  (ddp and fsdp)
            # bf16 (fsdp)
            scaled_loss.backward()

    info_dict['loss_dict']['loss'] = scaled_loss
    for loss_name, loss_value in info_dict['loss_dict'].items():
        if loss_value is not None:
            info_dict['loss_dict'][loss_name] = tensor_to_scalar(loss_value)

    return info_dict


def update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict):
    rank = int(os.environ.get('RANK', 0))
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1)
    use_amp = info_dict.get('use_amp', False)
    clip = info_dict.get('grad_clip', 50.0)
    batch_idx = info_dict["batch_idx"]
    if use_amp:
        assert scaler is not None

    grad_norm = 0.0
    if train_engine == "deepspeed":
        # NOTE(xcsong): The step() function in DeepSpeed engine updates the
        #   model parameters as well as the learning rate.
        #   Zeroing the gradients is handled automatically by
        #   DeepSpeed after the weights have been updated using a mini-batch.
        #   DeepSpeed also performs gradient averaging automatically at the
        #   gradient accumulation boundaries and addresses clip_grad_norm internally.
        #   `ds_model.step() =  clip_grad_norm_() + optimizer.step()
        #                       + optimizer.zero_grad() + scheduler.step()`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        info_dict["is_gradient_accumulation_boundary"] = \
            model.is_gradient_accumulation_boundary()
        model.step()
        grad_norm = model.get_global_grad_norm()
    elif (batch_idx + 1) % accum_grad == 0:
        # Use mixed precision training
        # fp16 (ddp fsdp)
        if scaler is not None:
            scaler.unscale_(optimizer)
            if train_engine == "torch_ddp":
                grad_norm = clip_grad_norm_(model.parameters(), clip)
            else:
                # fsdp
                grad_norm = model.clip_grad_norm_(clip)
            # Must invoke scaler.update() if unscale_() is used in
            # the iteration to avoid the following error:
            #   RuntimeError: unscale_() has already been called
            #   on this optimizer since the last update().
            # We don't check grad here since that if the gradient
            # has inf/nan values, scaler.step will skip
            # optimizer.step().
            scaler.step(optimizer)
            scaler.update()
        else:
            if train_engine == "torch_ddp":
                grad_norm = clip_grad_norm_(model.parameters(), clip)
            else:
                grad_norm = model.clip_grad_norm_(clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    info_dict["lrs"] = [group['lr'] for group in optimizer.param_groups]
    info_dict["grad_norm"] = tensor_to_scalar(grad_norm)

    return info_dict


def log_per_step(writer, info_dict, timer: Optional[StepTimer] = None):
    tag = info_dict["tag"]
    step = info_dict["step"]
    batch_idx = info_dict["batch_idx"]
    loss_dict = info_dict['loss_dict']
    epoch = info_dict.get('epoch', 0)
    train_engine = info_dict.get("train_engine", "torch_ddp")
    accum_grad = info_dict.get('accum_grad', 1) if tag != "CV" else 1
    log_interval = info_dict.get('log_interval', 10)
    lrs = info_dict.get("lrs", [0.0])
    is_gradient_accumulation_boundary = info_dict.get(
        "is_gradient_accumulation_boundary", False)

    rank = int(os.environ.get('RANK', 0))
    # TRAIN Tensorboard
    if tag == "TRAIN" and rank == 0 and writer is not None:
        if (train_engine == "deepspeed" and is_gradient_accumulation_boundary
            ) or (train_engine in ["torch_ddp", "torch_fsdp"] and
                  (batch_idx + 1) % accum_grad == 0):
            writer.add_scalar('train/train_loss',
                              tensor_to_scalar(loss_dict['loss']) * accum_grad,
                              step)
            writer.add_scalar('train/grad_norm', info_dict['grad_norm'], step)
            for name, value in loss_dict.items():
                if name != 'loss' and value is not None:
                    writer.add_scalar('train/{}'.format(name),
                                      tensor_to_scalar(value), step)
            # lr
            for i, lr in enumerate(lrs):
                writer.add_scalar('train/lr_{}'.format(i), lr, step)
    # CV Tensorboard
    elif "step_" in tag and rank == 0 and writer is not None:
        for name, value in loss_dict.items():
            writer.add_scalar('cv/{}'.format(name), tensor_to_scalar(value),
                              step)
        logging.info(
            'Epoch {} Step {} CV info lr {} cv_loss {} rank {} acc {}'.format(
                epoch, step + 1, lrs_to_str(lrs),
                tensor_to_scalar(loss_dict["loss"]), rank,
                tensor_to_scalar(loss_dict["acc"])))
        return

    # TRAIN & CV, Shell log (stdout)
    if (batch_idx + 1) % log_interval == 0:
        log_str = '{} | '.format(tag)
        if timer is not None:
            timer_step = step
            if info_dict.get("cv_step", None) is not None:
                timer_step = info_dict['cv_step']
            steps_per_second = timer.steps_per_second(timer_step)
            log_str += 'steps/sec {:.3f}| '.format(steps_per_second)
        log_str += 'Batch {}/{} loss {:.6f} '.format(
            epoch, batch_idx + 1 if 'save_interval' not in info_dict else
            (step + 1) * accum_grad,
            tensor_to_scalar(loss_dict['loss']) * accum_grad)
        for name, value in loss_dict.items():
            if name != 'loss' and value is not None:
                log_str += '{} {:.6f} '.format(name, tensor_to_scalar(value))
        if tag == "TRAIN":
            log_str += 'lr {} grad_norm {:.6f} rank {}'.format(
                lrs_to_str(lrs), info_dict['grad_norm'], rank)
        logging.debug(log_str)


def log_per_epoch(writer, info_dict):
    epoch = info_dict["epoch"]
    loss_dict = info_dict["loss_dict"]
    lrs = info_dict['lrs']
    rank = int(os.environ.get('RANK', 0))
    step = info_dict["step"]
    logging.info(
        'Epoch {} Step {} CV info lr {} cv_loss {} rank {} acc {}'.format(
            epoch, step, lrs_to_str(lrs), tensor_to_scalar(loss_dict["loss"]),
            rank, tensor_to_scalar(loss_dict["acc"])))

    if int(os.environ.get('RANK', 0)) == 0:
        for i, lr in enumerate(info_dict["lrs"]):
            writer.add_scalar('epoch/lr_{}'.format(i), lr, epoch)
        for name, value in loss_dict.items():
            writer.add_scalar('epoch/{}'.format(name), tensor_to_scalar(value),
                              epoch)


def freeze_modules(model, args):
    for name, param in model.named_parameters():
        for module_name in args.freeze_modules:
            if module_name in name:
                param.requires_grad = False
                logging.debug("{} module is freezed".format(name))


def reinit_lora(model, args, configs, tokenizer, seed=777):
    from tqdm import tqdm
    from wenet.finetune.lora.utils import estimate_gradient, reinit_lora_modules
    from wenet.finetune.lora.layers import LoRALayer
    from types import SimpleNamespace

    logging.info("reinit lora modules.")
    with open(args.lora_init_yaml, 'r') as file:
        lora_config = yaml.safe_load(file)

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset_conf = copy.deepcopy(configs['dataset_conf'])
    dataset_conf['batch_conf']['batch_size'] = lora_config['init_batch_size']
    dataset_type = configs.get('dataset', 'asr')
    dataset = init_dataset(dataset_type, args.data_type, args.train_data,
                           tokenizer, dataset_conf, True)
    dataloader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=args.pin_memory,
                            num_workers=args.num_workers,
                            persistent_workers=True,
                            generator=generator,
                            prefetch_factor=args.prefetch)
    additional_kwargs = {}
    if lora_config["init_config"]["mode"] == "gradient":
        named_grads = estimate_gradient(model, dataloader,
                                        lora_config['init_iters'])
        additional_kwargs["named_grads"] = named_grads
    lora_config = SimpleNamespace(**lora_config["init_config"])
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        if isinstance(module, LoRALayer):
            reinit_lora_modules(name, module, lora_config, **additional_kwargs)
    # lora_init_model needs to be saved, w0 = w0 - A0 * B0
    save_checkpoint(model, os.path.join(args.model_dir, "lora_init.pt"),
                    infos={"tag": "lora_init", **configs})
