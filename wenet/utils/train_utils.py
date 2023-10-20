# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Horizon Inc. (authors: Xingchen Song)
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

import copy
import datetime
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
from deepspeed.runtime.zero.stage_1_and_2 import (
    estimate_zero2_model_states_mem_needs_all_live
)
from deepspeed.runtime.zero.stage3 import (
    estimate_zero3_model_states_mem_needs_all_live
)
from deepspeed.utils.zero_to_fp32 import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import save_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing


def add_model_args(parser):
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument("--enc_init",
                        default=None,
                        type=str,
                        help="Pre-trained model to initialize encoder")
    parser.add_argument("--enc_init_mods",
                        default="encoder.",
                        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
                        help="List of encoder modules \
                        to initialize ,separated by a comma")
    parser.add_argument('--lfmmi_dir',
                        default='',
                        required=False,
                        help='LF-MMI dir')
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


def add_ddp_args(parser):
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
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
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser = deepspeed.add_config_arguments(parser)
    return parser


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    if args.train_engine == "torch_ddp":
        logging.info('training on multiple gpus, this gpu {}'.format(local_rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=world_size,
                                rank=rank)
    elif args.train_engine == "deepspeed":
        deepspeed.init_distributed(dist_backend=args.dist_backend,
                                   init_method=args.init_method,
                                   world_size=world_size,
                                   rank=rank)
    else:
        logging.info("Do nothing for cpu training")
    return world_size, local_rank, rank


def check_modify_and_save_config(args, configs):
    if args.train_engine == "torch_ddp":
        if args.use_amp:
            configs["dtype"] = "fp16"
        else:
            configs["dtype"] = "fp32"
    elif args.train_engine == "deepspeed":
        # NOTE(xcsong): DeepSpeed does not support uneven data. When using custom
        #   dataset, we need to manually ensure that the data is evenly distributed
        #   across all processe. we impl `tools/filter_uneven_data.py` for this func
        #   ref: https://github.com/microsoft/DeepSpeed/issues/2223
        #
        # NOTE(xsong):  We also need to keep
        #       `train_micro_batch_size_per_gpu == 1`
        #   and
        #       `accum_grad (in train_confomrer.yaml)
        #           == gradient_accumulation_steps (in ds_config.json)`
        #   The reason for such consistence checking lies in that deepspeed's
        #   dataloader uses PyTorch's torch.utils.data.DistributedSampler which does
        #   not support IterableDataset, IterableDataset is extremly useful in large
        #   scale training because it lets you stream the data without having to
        #   download the complete dataset.
        #   ref: https://github.com/microsoft/DeepSpeed/issues/1371
        #        https://github.com/microsoft/DeepSpeed/issues/285
        #
        #   To make deepspeed training compatible with IterableDataset, we have to
        #   use custom dataloader instead of deepspeed's native loader and thus we
        #   should configure batchsize in train_confomrer.yaml instead of
        #   ds_config.json. On the contrary, gradient accumulation steps should be
        #   configured in ds_config.json since it will be handled by deepspeed.
        #   ref: https://github.com/microsoft/DeepSpeed/issues/62
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs["dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs["dtype"] = "bf16"
        else:
            configs["dtype"] = "fp32"
        assert configs['dataset_conf']['batch_conf']['batch_type'] == "static"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        assert ds_configs["gradient_accumulation_steps"] == configs['accum_grad']

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    symbol_table = read_symbol_table(args.symbol_table)
    vocab_size = len(symbol_table)

    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    configs['lfmmi_dir'] = args.lfmmi_dir

    configs['train_engine'] = args.train_engine
    configs['use_amp'] = args.use_amp

    # Save configs to model_dir/train.yaml for inference and export
    if int(os.environ.get('RANK', 0)) == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    return configs


def init_dataset_and_dataloader(args, configs):
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False

    symbol_table = read_symbol_table(args.symbol_table)
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    train_dataset = Dataset(args.data_type, args.train_data, symbol_table,
                            train_conf, args.bpe_model, non_lang_syms, True)
    cv_dataset = Dataset(args.data_type,
                         args.cv_data,
                         symbol_table,
                         cv_conf,
                         args.bpe_model,
                         non_lang_syms,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)
    return train_dataset, cv_dataset, train_data_loader, cv_data_loader


def wrap_cuda_model(args, model):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    # TODO(xcsong): could one GPU use ddp? and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if args.train_engine == "torch_ddp":  # native pytorch ddp
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
        if args.fp16_grad_sync:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default as comm_hooks,
            )
            model.register_comm_hook(
                state=None, hook=comm_hooks.fp16_compress_hook
            )
    elif args.train_engine == "deepspeed":  # deepspeed
        # NOTE(xcsong): look in detail how the memory estimator API works:
        #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
        if int(os.environ.get('RANK', 0)) == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model, num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus_per_node=local_world_size,
                num_nodes=world_size // local_world_size)
        device = None     # Init device later
        pass              # Init DeepSpeed later
    else:
        device = torch.device('cpu')
        model = model.to(device)

    return model, device


def init_optimizer_and_scheduler(args, infos, configs, model):
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['optim_conf'])
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
            args=args, model=model, optimizer=optimizer,
            lr_scheduler=scheduler, model_parameters=model.parameters())

    step = infos.get('step', -1)
    scheduler.set_step(step)
    return model, optimizer, scheduler


def trace_and_print_model(args, model, enable_trace=True, enable_print=True):
    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if int(os.environ.get('RANK', 0)) == 0:
        if enable_trace:
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(args.model_dir, 'init.zip'))
        if enable_print:
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


def save_model(args, model, tag, infos):
    rank = int(os.environ.get('RANK', 0))
    if args.train_engine == "deepspeed":
        # NOTE(xcsong): All ranks should call this API, but only rank 0
        #   save the general model params. see:
        #   https://github.com/microsoft/DeepSpeed/issues/2993
        with torch.no_grad():
            model.save_checkpoint(save_dir=args.model_dir,
                                  tag=tag, client_state=infos)
            if args.save_states == "model_only" and rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    args.model_dir, "{}/{}.pt".format(args.model_dir, tag),
                    tag=tag)
                os.system("rm -rf {}/{}".format(args.model_dir, tag))
    elif rank == 0:
        # NOTE(xcsong): For torch_ddp & torch_cpu,
        #   only rank-0 should call this.
        save_model_path = os.path.join(args.model_dir, '{}.pt'.format(tag))
        save_checkpoint(model, save_model_path, infos)


def wenet_join(configs, device, group_join):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    train_engine = configs.get('train_engine', "torch_ddp")

    if train_engine != "deepspeed":
        return False

    try:
        # NOTE(xcsong): Why we need a new group?
        #   Because Deepspeed has its own group where all the relevant communication
        #   operations are executed. If we add a communication operation that is not
        #   managed by Deepspeed in this group, it's highly likely to cause
        #   communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join,
                               timeout=datetime.timedelta(seconds=30))
    except RuntimeError as e:
        logging.info("Detected uneven workload distribution: {}\n".format(e) +
                     "Break current worker to manually join all workers, " +
                     "world_size {}, current rank {}, current local_rank {}".format(
                         world_size, rank, local_rank))
        return True

    return False


def batch_forward(configs, model, batch, scaler):
    train_engine = configs.get('train_engine', "torch_ddp")
    accum_grad = configs.get('accum_grad', 1)

    dtype = configs.get("dtype", "fp32")
    if dtype == "fp16":
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    else:  # fp32
        dtype = None

    if train_engine == "deepspeed":
        # deepspeed
        with torch.cuda.amp.autocast(
            enabled=dtype is not None, dtype=dtype, cache_enabled=False
        ):
            loss_dict = model(batch["feats"], batch["feats_lengths"],
                              batch["target"], batch["target_lengths"])
    else:
        # torch_ddp or torch_cpu
        # autocast context
        # The more details about amp can be found in
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        with torch.cuda.amp.autocast(scaler is not None):
            loss_dict = model(batch["feats"], batch["feats_lengths"],
                              batch["target"], batch["target_lengths"])
    loss_dict['loss'] = loss_dict['loss'] / accum_grad

    return loss_dict


def batch_backward(configs, model, loss_dict, scaler):
    train_engine = configs.get("train_engine", "torch_ddp")
    use_amp = configs.get('use_amp', False)
    if use_amp:
        assert scaler is not None
    loss = loss_dict['loss']

    if train_engine == "deepspeed":  # deepspeed
        # NOTE(xcsong): Zeroing the gradients is handled automatically by
        #   DeepSpeed after the weights have been updated using a mini-batch.
        #   DeepSpeed also performs gradient averaging automatically at the
        #   gradient accumulation boundaries and addresses clip_grad_norm
        #   internally. In other words, `model.backward(loss)` is equivalent to
        #   `loss.backward() + clip_grad_norm_()
        #                    + optimizer.zero_grad() + accum_grad`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        model.backward(loss)
    else:             # torch_ddp or torch_cpu
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()


def update_parameter_and_lr(
    configs, model, optimizer,
    scheduler, scaler, info_dict
):
    train_engine = configs.get("train_engine", "torch_ddp")
    accum_grad = configs.get('accum_grad', 1)
    use_amp = configs.get('use_amp', False)
    clip = configs.get('grad_clip', 50.0)
    rank = int(os.environ.get('RANK', 0))
    batch_idx = info_dict["batch_idx"]
    if use_amp:
        assert scaler is not None

    if train_engine == "deepspeed":
        # NOTE(xcsong): The step() function in DeepSpeed engine updates the
        #   model parameters as well as the learning rate. There is no need
        #   to manually perform scheduler.step(). In other words:
        #   `ds_model.step() = optimizer.step() + scheduler.step()`
        #   ref: https://www.deepspeed.ai/tutorials/megatron/#using-the-training-api
        model.step()
        info_dict["is_gradient_accumulation_boundary"] = \
            model.is_gradient_accumulation_boundary()
    elif batch_idx % accum_grad == 0 and batch_idx != 0:
        # Use mixed precision training
        if use_amp:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), clip)
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
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    info_dict["lr"] = optimizer.param_groups[0]['lr']

    return info_dict


def log_per_step(configs, loss_dict, info_dict, writer, tag):
    epoch = configs.get('epoch', 0)
    train_engine = configs.get("train_engine", "torch_ddp")
    accum_grad = configs.get('accum_grad', 1) if tag == "TRAIN" else 1
    log_interval = configs.get('log_interval', 10)

    loss = loss_dict['loss']
    rank = int(os.environ.get('RANK', 0))

    batch_idx = info_dict["batch_idx"]
    lr = info_dict.get("lr", 0.0)
    history_loss = info_dict.get("history_loss", 0.0)
    step = info_dict.get("step", -1)
    is_gradient_accumulation_boundary = info_dict.get(
        "is_gradient_accumulation_boundary", False)

    if tag == "TRAIN":
        if train_engine == "deepspeed":
            if rank == 0 and writer is not None \
                    and is_gradient_accumulation_boundary:
                writer.add_scalar('train_loss', loss.item(), step)
        elif batch_idx % accum_grad == 0 and batch_idx != 0:
            if rank == 0 and writer is not None:
                writer.add_scalar('train_loss', loss.item(), step)

    if batch_idx % log_interval == 0:
        log_str = '{} Batch {}/{} loss {:.6f} '.format(
            tag, epoch, batch_idx,
            loss.item() * accum_grad)
        for name, value in loss_dict.items():
            if name != 'loss' and value is not None:
                log_str += '{} {:.6f} '.format(name, value.item())
        if tag == "TRAIN":
            log_str += 'lr {:.8f} rank {}'.format(lr, rank)
        elif tag == "CV":
            log_str += 'history loss {:.6f} rank {}'.format(history_loss, rank)
        logging.debug(log_str)


def log_per_epoch(args, info_dict, writer, tag):
    epoch = info_dict["epoch"]
    if int(os.environ.get('RANK', 0)) == 0:
        writer.add_scalar('epoch/cv_loss', info_dict["cv_loss"], epoch)
        writer.add_scalar('epoch/lr', info_dict["lr"], epoch)
        with open("{}/{}.yaml".format(args.model_dir, epoch), 'w') as fout:
            data = yaml.dump(info_dict)
            fout.write(data)
