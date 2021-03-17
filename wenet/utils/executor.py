# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue
            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if is_distributed and batch_idx % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            with context():
                loss, loss_att, l_loss_att, r_loss_att, loss_ctc = model(
                    feats, feats_lengths, target, target_lengths)
                loss = loss / accum_grad
                loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1

            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    batch_idx, num_total_batch,
                    loss.item() * accum_grad)
                if loss_att is not None:
                    log_str += 'loss_att {:.6f} '.format(loss_att.item())
                if l_loss_att is not None:
                    log_str += 'l_loss_att {:.6f} '.format(l_loss_att.item())
                if r_loss_att is not None:
                    log_str += 'r_loss_att {:.6f} '.format(r_loss_att.item())
                if loss_ctc is not None:
                    log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                logging.debug(log_str)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        num_seen_utts = 0
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, l_loss_att, r_loss_att, loss_ctc = model(
                    feats, feats_lengths, target, target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        batch_idx, num_total_batch, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if l_loss_att is not None:
                        log_str += 'l_loss_att {:.6f} '.format(
                            l_loss_att.item())
                    if r_loss_att is not None:
                        log_str += 'r_loss_att {:.6f} '.format(
                            r_loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    logging.debug(log_str)

        return total_loss, num_seen_utts
