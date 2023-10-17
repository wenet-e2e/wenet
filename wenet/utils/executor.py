# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch

from wenet.utils.train_utils import (batch_forward, batch_backward,
                                     update_parameter_and_lr, log_per_step)


class Executor:

    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        accum_grad = args.get('accum_grad', 1)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(accum_grad))

        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch

                batch_dict = {}
                batch_dict["feats"] = feats.to(device)
                batch_dict["target"] = target.to(device)
                batch_dict["feats_lengths"] = feats_lengths.to(device)
                batch_dict["target_lengths"] = target_lengths.to(device)

                if target_lengths.size(0) == 0:
                    continue

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if args.get("train_engine", "torch_ddp") == "torch_ddp" and \
                        batch_idx % accum_grad != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    loss_dict = batch_forward(args, model, batch_dict, scaler)
                    batch_backward(args, model, loss_dict, scaler)

                info_dict = {"batch_idx": batch_idx, "step": self.step}

                info_dict["lr"] = update_parameter_and_lr(
                    args, model, optimizer, scheduler,
                    scaler, info_dict
                )

                log_per_step(args, loss_dict, info_dict, writer, tag="TRAIN")

                self.step += 1

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch

                batch_dict = {}
                batch_dict["feats"] = feats.to(device)
                batch_dict["target"] = target.to(device)
                batch_dict["feats_lengths"] = feats_lengths.to(device)
                batch_dict["target_lengths"] = target_lengths.to(device)

                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue

                loss_dict = batch_forward(args, model, batch_dict, None)
                loss = loss_dict['loss']

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts

                info_dict = {"batch_idx": batch_idx, "history_loss": total_loss / num_seen_utts}  # noqa
                log_per_step(args, loss_dict, info_dict, writer, tag="CV")
        return total_loss, num_seen_utts
