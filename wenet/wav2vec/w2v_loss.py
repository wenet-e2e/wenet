#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# The origial Wav2vec work is in:
# Paper: https://arxiv.org/pdf/2006.11477.pdf
# Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec


from distutils.version import LooseVersion
import logging

import numpy as np
import six
import torch
import torch.nn.functional as F
import math

class W2vLoss(torch.nn.Module):
    def __init__(self,infonce=False,loss_weights=None):
        super().__init__()
        self.infonce = infonce
        self.loss_weights = loss_weights

    def forward(self, model, net_output ,reduce=True):
  
        losses = []
        weights = None
        self.infonce=True

        logits = model.get_logits(net_output)
        target = model.get_targets(None, net_output)
        sample_size = target.numel() 

        if self.infonce:
            loss = F.cross_entropy(
                logits,
                target,
                reduction="sum" if reduce else "none",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),
                weights,
                reduction="sum" if reduce else "none",
            )

        losses.append(loss.detach().clone())

        if self.loss_weights is not None:    
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
            self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)
                
        logging_output = {
            "loss": loss.item()/sample_size / math.log(2) if reduce else loss,
            "sample_size": sample_size,
        }

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss,sample_size,logging_output

