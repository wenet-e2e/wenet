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

import copy
import datetime
import logging
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch

from wenet.utils.train_utils import (wenet_join, batch_forward, batch_backward,
                                     update_parameter_and_lr, log_per_step,
                                     save_model)


class Executor:

    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, train_data_loader,
              cv_data_loader, writer, configs, scaler, group_join):
        ''' Train one epoch
        '''
        model.train()
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                if wenet_join(group_join, info_dict):
                    break

                if batch_dict["target_lengths"].size(0) == 0:
                    continue

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict.get("train_engine", "torch_ddp") == "torch_ddp" and \
                        (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler,
                                              info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer,
                                                    scheduler, scaler,
                                                    info_dict)
                save_interval = info_dict.get('save_interval', 100000000000000)
                if self.step % save_interval == 0 and self.step != 0 \
                        and (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    total_loss, num_seen_utts = self.cv(
                        model, cv_data_loader, configs)
                    model.train()
                    info_dict.update({
                        "tag":
                        "step_{}".format(self.step),
                        "cv_loss":
                        total_loss / num_seen_utts,
                        "save_time":
                        datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "lr":
                        optimizer.param_groups[0]['lr']
                    })
                    save_model(model, info_dict)
                log_per_step(writer, info_dict)
                self.step += 1 if (batch_idx +
                                   1) % info_dict["accum_grad"] == 0 else 0

    def cv(self, model, cv_data_loader, configs):
        ''' Cross validation on
        '''
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, total_loss = 1, 0.0  # in order to avoid division by 0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = "CV"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx

                num_utts = batch_dict["target_lengths"].size(0)
                if num_utts == 0:
                    continue

                info_dict = batch_forward(model, batch_dict, None, info_dict)
                loss = info_dict['loss_dict']['loss']

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts

                info_dict["history_loss"] = total_loss / num_seen_utts
                log_per_step(writer=None, info_dict=info_dict)
        return total_loss, num_seen_utts
