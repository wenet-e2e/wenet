# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
# from contextlib import nullcontext
# if your python version < 3.7 use the below one
import torch
import time


class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)

        for batch_idx, batch in enumerate(data_loader):
            start = time.perf_counter()
            feats, feats_lengths, target, target_lengths = batch
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue
            loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                             target_lengths)
            scheduler.step()
            model.setOptimizer(optimizer)
            self.step += 1
            end = time.perf_counter()
            writer.add_scalar('train_loss', loss.mean().item(), self.step)
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    epoch, batch_idx,
                    loss.mean().item())
                if loss_att is not None:
                    log_str += 'loss_att {:.6f} '.format(loss_att.mean().item())
                if loss_ctc is not None:
                    log_str += 'loss_ctc {:.6f} '.format(loss_ctc.mean().item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                log_str += ' throughput {:.0f} '.format(
                    num_utts / (end - start))
                logging.debug(log_str)


    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        rank = args.get('rank', 0)
        epoch = args.get('epoch', 0)
        # log_interval = args.get('log_interval', 10)
        log_interval = 1
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                feats, feats_lengths, target, target_lengths = batch
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss.mean()):
                    num_seen_utts += num_utts
                    total_loss += loss.mean().item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        epoch, batch_idx, loss.mean().item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(
                            loss_att.mean().item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(
                            loss_ctc.mean().item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    log_str += ' rank {}'.format(rank)
                    logging.debug(log_str)
        return total_loss, num_seen_utts
