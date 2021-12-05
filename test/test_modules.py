#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Label smoothing module."""

import numpy as np
import os
import pdb

import torch

from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss


def test_label_smoothing():
    """Test the accuracy of label smoothing loss."""
    # load input samples
    example_dir = "/home/work_nfs4_ssd/pcguo/code/wenet/test/examples/label_smoothing"
    x = torch.from_numpy(np.load(os.path.join(example_dir, "x.npy")))
    target = torch.from_numpy(np.load(os.path.join(example_dir, "target.npy"))).long()

    # define the loss object
    criterion = LabelSmoothingLoss(
        size=4233,
        padding_idx=-1,
        smoothing=0.1,
        normalize_length=False,
    )
    kl_loss = criterion(x, target)
    print(kl_loss)


if __name__ == "__main__":
    test_label_smoothing()
