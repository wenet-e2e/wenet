# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

from __future__ import print_function

import argparse
import copy
import datetime
import deepspeed
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live  # noqa
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live  # noqa
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.checkpoint import (load_checkpoint, save_checkpoint,
                                    load_trained_modules)
from wenet.utils.executor import Executor
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.scheduler import WarmupLR, NoamHoldAnnealing
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='global rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='''number of total processes/gpus for
                        distributed training''')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--ddp.init_method',
                        dest='init_method',
                        default=None,
                        help='ddp init method')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--fp16_grad_sync',
                        action='store_true',
                        default=False,
                        help='Use fp16 gradient sync for ddp')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
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

    # Begin deepspeed related config
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    # End deepspeed related config


    # DeepSpeed automaticly add '--deepspeed' and '--deepspeed_config' to parser
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # NOTE(xcsong): deepspeed set CUDA_VISIBLE_DEVICES internally
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) if not args.deepspeed \
        else os.environ['CUDA_VISIBLE_DEVICES']

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)
    if args.deepspeed:
        with open(args.deepspeed_config, 'r') as fin:
            ds_configs = json.load(fin)
        if "fp16" in ds_configs and ds_configs["fp16"]["enabled"]:
            configs["ds_dtype"] = "fp16"
        elif "bf16" in ds_configs and ds_configs["bf16"]["enabled"]:
            configs["ds_dtype"] = "bf16"
        else:
            configs["ds_dtype"] = "fp32"

    # deepspeed read world_size from env
    if args.deepspeed:
        assert args.world_size == -1
    # distributed means pytorch native ddp, it parse world_size from args
    distributed = args.world_size > 1
    local_rank = args.rank
    world_size = args.world_size
    if distributed:
        logging.info('training on multiple gpus, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=world_size,
                                rank=local_rank)
    elif args.deepspeed:
        # Update local_rank & world_size from enviroment variables
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        deepspeed.init_distributed(dist_backend=args.dist_backend,
                                   init_method=args.init_method,
                                   rank=local_rank,
                                   world_size=world_size)

    symbol_table = read_symbol_table(args.symbol_table)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

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
    if args.deepspeed:
        assert train_conf['batch_conf']['batch_type'] == "static"
        assert ds_configs["train_micro_batch_size_per_gpu"] == 1
        configs['accum_grad'] = ds_configs["gradient_accumulation_steps"]
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

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']
    vocab_size = len(symbol_table)

    # Save configs to model_dir/train.yaml for inference and export
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = args.cmvn
    configs['is_json_cmvn'] = True
    configs['lfmmi_dir'] = args.lfmmi_dir

    if local_rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # Init asr model from configs
    model = init_model(configs)
    print(model) if local_rank == 0 else None
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {:,d}'.format(num_params)) if local_rank == 0 else None  # noqa

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if local_rank == 0:
        script_model = torch.jit.script(model)
        script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    elif args.enc_init is not None:
        logging.info('load pretrained encoders: {}'.format(args.enc_init))
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:  # native pytorch ddp
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
    elif args.deepspeed:  # deepspeed
        # NOTE(xcsong): look in detail how the memory estimator API works:
        #   https://deepspeed.readthedocs.io/en/latest/memory.html#discussion
        if local_rank == 0:
            logging.info("Estimating model states memory needs (zero2)...")
            estimate_zero2_model_states_mem_needs_all_live(
                model, num_gpus_per_node=world_size, num_nodes=1)
            logging.info("Estimating model states memory needs (zero3)...")
            estimate_zero3_model_states_mem_needs_all_live(
                model, num_gpus_per_node=world_size, num_nodes=1)
        device = None     # Init device later
        pass              # Init DeepSpeed later
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

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
    if args.deepspeed:
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

    final_epoch = None
    configs['rank'] = local_rank
    configs['is_distributed'] = distributed   # pytorch native ddp
    configs['is_deepspeed'] = args.deepspeed  # deepspeed
    configs['use_amp'] = args.use_amp
    if args.deepspeed and start_epoch == 0:
        # NOTE(xcsong): All ranks should call this API, but only rank 0
        #   save the general model params. see:
        #   https://github.com/microsoft/DeepSpeed/issues/2993
        with torch.no_grad():
            model.save_checkpoint(save_dir=model_dir, tag='init')
            if args.save_states == "model_only" and local_rank == 0:
                convert_zero_checkpoint_to_fp32_state_dict(
                    model_dir, "{}/init.pt".format(model_dir), tag='init')
                os.system("rm -rf {}/{}".format(model_dir, "init"))
    elif not args.deepspeed and start_epoch == 0 and local_rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    # used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        device = model.local_rank if args.deepspeed else device
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)
        cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        infos = {
            'epoch': epoch, 'lr': lr, 'cv_loss': cv_loss, 'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        if local_rank == 0:
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
            with open("{}/{}.yaml".format(model_dir, epoch), 'w') as fout:
                data = yaml.dump(infos)
                fout.write(data)
        if args.deepspeed:
            # NOTE(xcsong): All ranks should call this API, but only rank 0
            #   save the general model params. see:
            #   https://github.com/microsoft/DeepSpeed/issues/2993
            with torch.no_grad():
                model.save_checkpoint(save_dir=model_dir,
                                      tag='{}'.format(epoch),
                                      client_state=infos)
                if args.save_states == "model_only" and local_rank == 0:
                    convert_zero_checkpoint_to_fp32_state_dict(
                        model_dir, "{}/{}.pt".format(model_dir, epoch),
                        tag='{}'.format(epoch))
                    os.system("rm -rf {}/{}".format(model_dir, epoch))
        elif not args.deepspeed and local_rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(model, save_model_path, infos)
        final_epoch = epoch

    if final_epoch is not None and local_rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
