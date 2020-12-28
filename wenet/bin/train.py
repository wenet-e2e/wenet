# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

from __future__ import print_function

import argparse
import copy
import logging
import os

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tensorboardX import SummaryWriter

from wenet.dataset.dataset import CollateFunc, AudioDataset
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.ctc import CTC
from wenet.transformer.asr_model import ASRModel
from wenet.utils.executor import Executor
from wenet.utils.scheduler import WarmupLR
from wenet.utils.checkpoint import save_checkpoint, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.rank',
                        dest='rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--ddp.world_size',
                        dest='world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
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
    parser.add_argument('--cmvn', default=None, help='global cmvn file')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Set random seed
    torch.manual_seed(777)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin)

    distributed = args.world_size > 1

    # Init dataset and data loader
    collate_func = CollateFunc(**configs['collate_conf'],
                               **configs['spec_aug_conf'],
                               cmvn=args.cmvn)
    cv_collate_conf = copy.copy(configs['collate_conf'])
    cv_collate_conf['spec_aug'] = False
    cv_collate_func = CollateFunc(**cv_collate_conf, cmvn=args.cmvn)
    dataset_conf = configs.get('dataset_conf', {})
    train_dataset = AudioDataset(args.train_data, **dataset_conf)
    cv_dataset = AudioDataset(args.cv_data, **dataset_conf)

    if distributed:
        logging.info('training on multiple gpu, this gpu {}'.format(args.gpu))
        dist.init_process_group(args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size,
                                rank=args.rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True)
        cv_sampler = torch.utils.data.distributed.DistributedSampler(
            cv_dataset, shuffle=False)
    else:
        train_sampler = None
        cv_sampler = None

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=collate_func,
                                   sampler=train_sampler,
                                   shuffle=(train_sampler is None),
                                   batch_size=1,
                                   num_workers=args.num_workers)
    cv_data_loader = DataLoader(cv_dataset,
                                collate_fn=cv_collate_func,
                                sampler=cv_sampler,
                                shuffle=False,
                                batch_size=1,
                                num_workers=args.num_workers)

    # Init transformer model
    input_dim = train_dataset.input_dim
    vocab_size = train_dataset.output_dim
    # Save configs to model_dir/train.yaml for inference and export
    if args.rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'train.yaml')
        with open(saved_config_path, 'w') as fout:
            configs['input_dim'] = input_dim
            configs['output_dim'] = vocab_size
            data = yaml.dump(configs)
            fout.write(data)

    encoder_type = configs.get('encoder', 'conformer')
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim, **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim, **configs['encoder_conf'])
    decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                 **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())

    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )

    print(model)

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    script_model = torch.jit.script(model)
    script_model.save(os.path.join(args.model_dir, 'init.zip'))
    executor = Executor()
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)

    num_epochs = configs.get('max_epoch', 100)
    model_dir = args.model_dir
    writer = None
    if args.rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if distributed:
        assert (torch.cuda.is_available())
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True)
        device = torch.device("cuda")
    else:
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    final_epoch = None
    configs['rank'] = args.rank
    if start_epoch == 0 and args.rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    executor.step = step
    scheduler.set_step(step)
    for epoch in range(start_epoch, num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs)
        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device, configs)
        if args.world_size > 1:
            # all_reduce expected a sequence parameter, so we use [num_seen_utts].
            num_seen_utts = torch.Tensor([num_seen_utts]).to(device)
            # the default operator in all_reduce function is sum.
            dist.all_reduce(num_seen_utts)
            total_loss = torch.Tensor([total_loss]).to(device)
            dist.all_reduce(total_loss)
            cv_loss = total_loss[0] / num_seen_utts[0]
            cv_loss = cv_loss.item()
        else:
            cv_loss = total_loss / num_seen_utts

        logging.info('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        if args.rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'cv_loss': cv_loss,
                'step': executor.step
            })
            writer.add_scalars('epoch', {'cv_loss': cv_loss, 'lr': lr}, epoch)
        final_epoch = epoch

    if final_epoch is not None and args.rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
