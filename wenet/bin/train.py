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
import datetime
import logging
import os
import torch
import yaml

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.train_utils import (add_model_args, add_dataset_args,
                                     add_ddp_args, add_deepspeed_args,
                                     init_distributed, init_dataset_and_dataloader,
                                     check_modify_and_save_config,
                                     init_optimizer_and_scheduler,
                                     trace_and_print_model, wrap_cuda_model,
                                     init_summarywriter, save_model,
                                     log_per_epoch)

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed', 'torch_cpu'],
                        help='Engine for paralleled training')
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_deepspeed_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


# On worker errors, this tool will summarize the details of the error
#   (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # Init env for ddp OR deepspeed
    world_size, local_rank, rank = init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Init asr model from configs
    infos, model = init_model(args, configs)

    # Check model is jitable & print model archtectures
    trace_and_print_model(args, model, enable_trace=True, enable_print=True)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    model, device = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, infos, configs, model)

    # Save checkpoints
    save_model(args, model, tag="init",
               infos={"save_time": datetime.datetime.now()
                      .strftime('%d/%m/%Y %H:%M:%S')})

    # Get executor
    executor = Executor()
    executor.step = infos.get('step', -1)

    # Init scaler, used for pytorch amp mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Start training loop
    start_epoch = infos.get('epoch', -1) + 1
    final_epoch = None
    for epoch in range(start_epoch, configs.get('max_epoch', 100)):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))

        device = model.local_rank if args.deepspeed else device

        # NOTE(xcsong): monitored barrier requires gloo process group to
        #   perform host-side sync. this group is used to join workers for
        #   deepspeed, more infos see `train_utils.py`
        group_join = dist.new_group(backend="gloo")

        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs, scaler, group_join)

        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)
        cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        infos = {
            'epoch': epoch, 'lr': lr, 'cv_loss': cv_loss, 'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }
        log_per_epoch(args, infos, writer, tag="")

        save_model(args, model, tag=str(epoch), infos=infos)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(args.model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()
