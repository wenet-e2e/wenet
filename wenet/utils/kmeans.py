import argparse
from functools import partial
import sys
import os
import copy
import torch
from torch.nn.utils.rnn import pad_sequence
import yaml
from wenet.dataset import processor

from wenet.dataset.datapipes import WenetRawDatasetSource
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.common import mask_to_bias
from wenet.utils.init_model import init_model
from wenet.utils.mask import make_pad_mask

from torch.utils.data import DataLoader

import torch.distributed as dist


def local_batched_bincount(samples: torch.Tensor, n_bins: int) -> torch.Tensor:
    groups, dtype, device = samples.size(0), samples.dtype, samples.device
    target = torch.zeros(groups, n_bins, dtype=dtype, device=device)
    values = torch.ones_like(samples, device=device, dtype=samples.dtype)
    target.scatter_add_(-1, samples, values)
    return target


def sample_multinomial(total_count, probs):
    device = probs.device

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long, device=device)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p
    return sample


def all_gather_variably_sized(x, sizes, dim=0):
    rank = dist.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(
            [size if i == dim else s for i, s in enumerate(x.size())])
        dist.broadcast(t, src=i, async_op=True)
        all_x.append(t)
    dist.barrier()
    return all_x


def sample_local_vectors(samples, num):

    def _generate_indices(input_tensor, N):
        B, device = input_tensor.shape[0], input_tensor.device
        if B >= N:
            indices = torch.randperm(B, device=device)[:N]
        else:
            indices = torch.randint(0, B, (N, ), device=device)
        return indices

    indices = _generate_indices(samples, num)
    return samples[indices]


def distributed_sample_vectors(local_samples, num):
    """
    Args:
        local_samples: [batch, dim]
    """
    batch_size, _ = local_samples.size()
    local_size = torch.tensor(batch_size,
                              dtype=torch.long,
                              device=local_samples.device)
    all_sizes = [
        torch.empty_like(local_size) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_sizes, local_size)
    global_size = torch.stack(all_sizes, dim=0)  # [global_rank, dim]

    rank = dist.get_rank()
    if rank == 0:
        samples_per_rank = sample_multinomial(num,
                                              global_size / global_size.sum())
    else:
        samples_per_rank = torch.empty_like(global_size)
    dist.broadcast(samples_per_rank, src=0)

    local_samples = sample_local_vectors(local_samples, samples_per_rank[rank])
    global_samples = all_gather_variably_sized(local_samples,
                                               samples_per_rank,
                                               dim=0)
    return torch.cat(global_samples, dim=0).unsqueeze(0)


class MinibatchKmeansClusterOneStep(torch.nn.Module):

    def __init__(self,
                 num_clusters: int,
                 dim: int,
                 num_groups: int = 1,
                 use_cosine_sim=False,
                 process_group=None) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.process_group = process_group
        self.dim = dim
        self.use_cosine_sim = use_cosine_sim
        self.groups = num_groups
        self.register_buffer('means', torch.randn(num_groups, num_clusters,
                                                  dim))
        self.register_buffer('means_old',
                             torch.randn(num_groups, num_clusters, dim))
        self.register_buffer('weight_sum', torch.zeros(num_groups,
                                                       num_clusters))

        self.is_initialized = False

    def compute_distance(self, samples: torch.Tensor,
                         means: torch.Tensor) -> torch.Tensor:
        if self.use_cosine_sim:
            distance = -(samples @ means.transpose(1, 2))
        else:
            distance = torch.sum(samples**2, dim=-1, keepdim=True) + torch.sum(
                means**2, dim=-1, keepdim=True).transpose(
                    1, 2) - 2 * (samples @ means.transpose(1, 2))
        return distance

    @torch.no_grad()
    @torch.jit.unused
    def _one_step(self, samples: torch.Tensor) -> torch.Tensor:
        """ One step iter:
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_k_means_minibatch.pyx#L87
        Args:
            samples: shape [groups, n_samples, dim]
        """
        process_group = dist.group.WORLD
        if self.process_group is not None:
            process_group = self.process_group
        distance = self.compute_distance(
            samples, self.means)  # [groups, n_samples, num_clusters]
        bins = torch.argmin(distance, dim=-1)  # [groups, n_samples]
        selected_distances = torch.gather(
            distance, 2, bins.unsqueeze(-1)).squeeze(-1)  # [groups, n_samples]
        inertiais = selected_distances.sqrt().sum()

        bins_count = local_batched_bincount(
            bins, self.num_clusters)  # [groups, n_clusters]
        dist.all_reduce(bins_count, group=process_group)
        bins_zeros = bins_count == 0
        new_means = torch.zeros_like(self.means)
        new_means.scatter_add_(1,
                               bins.unsqueeze(2).repeat(1, 1, self.dim),
                               samples)
        dist.all_reduce(new_means, group=process_group)
        new_means += self.means * self.weight_sum.unsqueeze(2)
        weight_sum = self.weight_sum + bins_count
        weight_sum_nonzero = torch.where(weight_sum == 0, 1., weight_sum)
        alpha = 1 / weight_sum_nonzero
        new_means *= alpha.unsqueeze(2)

        new_means = torch.where(bins_zeros.unsqueeze(2), self.means, new_means)
        if self.use_cosine_sim:
            new_means = torch.nn.functional.normalize(new_means, p=2, dim=-1)
        self.means.copy_(new_means)
        self.weight_sum.copy_(weight_sum)

        return inertiais

    @torch.jit.unused
    @torch.no_grad()
    def forward(self, input: torch.Tensor):
        """ forward for training
        Args
            input: shape [num_samples, dim]

        Returns:
            means: shape [groups, num_clusters, dim]
        """
        if not self.is_initialized:
            means = distributed_sample_vectors(input, self.num_clusters)
            self.means.copy_(means)
            self.is_initialized = True
        inertia = self._one_step(input.unsqueeze(0))

        return {"means": self.means, "inertias": inertia}

    @torch.jit.unused
    def encode(self, input: torch.Tensor):
        """ encoder input to ids
        Args:
            input: shape [num_samples, dim]

        Returns:
            indices: shape [num_groups, n_samples]
        """
        distance = self.compute_distance(input, self.means)
        bins = torch.argmin(distance, dim=-1)  # [groups, n_samples]
        return bins


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='wav data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp32',
                        choices=['fp16', 'fp32', 'bf16'],
                        help='model\'s dtype')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--max_iter',
                        default=1000,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--num_cluster',
                        default=8196,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='speech encoder to extract features')
    parser.add_argument('--kmeans_model_dir',
                        type=str,
                        help='speech encoder to extract features')
    parser.add_argument('--prefetch',
                        type=int,
                        default=10,
                        help='speech encoder to extract features')
    parser.add_argument('--save_interval',
                        type=int,
                        default=1000,
                        help='speech encoder to extract features')

    args = parser.parse_args()
    print(args)
    return args


class EmbeddingEncoder(torch.nn.Module):

    def __init__(self, encoder: BaseEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    @torch.no_grad()
    def forward(self, input: torch.Tensor, input_lens: torch.Tensor,
                n_layers: int):
        xs = input
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        masks = ~make_pad_mask(input_lens, input.size(1)).unsqueeze(
            1)  # (B, 1, T)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        masks_pad = masks
        if self.encoder.use_sdpa:
            masks = mask_to_bias(masks, xs.dtype)
        for (i, layer) in enumerate(self.encoder.encoders):
            if i == n_layers:
                break
            xs, masks, _, _ = layer(xs, masks, pos_emb, masks_pad)
        if n_layers == len(
                self.encoder.encoders) and self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)

        return xs, masks


def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "feats_lengths": feats_lengths,
    }
    return batch


def get_dataset(configs, data_list_file, partition, args):
    conf = copy.deepcopy(configs['dataset_conf'])
    conf['filter_conf']['max_length'] = 102400
    conf['filter_conf']['min_length'] = 0
    conf['filter_conf']['token_max_length'] = 102400
    conf['filter_conf']['token_min_length'] = 0
    conf['filter_conf']['max_output_input_ratio'] = 102400
    conf['filter_conf']['min_output_input_ratio'] = 0
    conf['speed_perturb'] = False
    conf['spec_aug'] = False
    conf['spec_sub'] = False
    conf['spec_trim'] = False
    conf['shuffle'] = False
    conf['sort'] = False
    conf['cycle'] = 100000000000
    conf['list_shuffle'] = False
    conf['fbank_conf']['dither'] = 0.0
    conf['batch_conf']['batch_type'] = "static"
    conf['batch_conf']['batch_size'] = 16

    cycle = conf.get('cycle', 1)
    # stage1 shuffle: source
    list_shuffle = conf.get('list_shuffle', True)
    list_shuffle_size = sys.maxsize
    if list_shuffle:
        list_shuffle_conf = conf.get('list_shuffle_conf', {})
        list_shuffle_size = list_shuffle_conf.get('shuffle_size',
                                                  list_shuffle_size)
    dataset = WenetRawDatasetSource(data_list_file,
                                    partition=partition,
                                    shuffle=list_shuffle,
                                    shuffle_size=list_shuffle_size,
                                    cycle=cycle)
    dataset = dataset.map(processor.parse_json)
    dataset = dataset.map_ignore_error(processor.decode_wav)

    filter_conf = conf.get('filter_conf', {})
    dataset = dataset.filter(partial(processor.filter, **filter_conf))
    resample_conf = conf.get('resample_conf', {})
    dataset = dataset.map(partial(processor.resample, **resample_conf))

    fbank_conf = conf.get('fbank_conf', {})
    dataset = dataset.map(
        partial(processor.compute_w2vbert_fbank, **fbank_conf))

    shuffle = conf.get('shuffle', True)
    if shuffle:
        shuffle_conf = conf.get('shuffle_conf', {})
        dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'])

    sort = conf.get('sort', True)
    if sort:
        sort_conf = conf.get('sort_conf', {})
        dataset = dataset.sort(buffer_size=sort_conf['sort_size'],
                               key_func=processor.sort_by_feats)

    batch_conf = conf.get('batch_conf', {})
    batch_size = batch_conf.get('batch_size', 16)
    dataset = dataset.batch(batch_size, wrapper_class=padding)
    generator = torch.Generator()
    generator.manual_seed(777)
    train_dataloader = DataLoader(dataset,
                                  batch_size=None,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  persistent_workers=True,
                                  generator=generator,
                                  prefetch_factor=args.prefetch)
    return dataset, train_dataloader


def init_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')


def main():
    args = get_args()
    assert torch.cuda.is_available()

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # init dist
    speech_model, _ = init_model(args, configs)
    speech_model.eval()
    embed_model = EmbeddingEncoder(speech_model.encoder)
    kmeans_model = MinibatchKmeansClusterOneStep(
        args.num_cluster,
        speech_model.encoder.output_size(),
        1,
    )
    embed_model.eval()
    kmeans_model.eval()
    init_distributed()
    embed_model.cuda()
    kmeans_model.cuda()
    device = torch.device('cuda')

    _, dataloader = get_dataset(configs, args.train_data, True, args)
    rank = dist.get_rank()
    with torch.no_grad():
        dist.barrier()
        for (i, batch) in enumerate(dataloader):
            if i > args.max_iter:
                break
            speech = batch['feats'].to(device)
            speech_lens = batch['feats_lengths'].to(device)
            encoder_out, encoder_out_mask = embed_model(speech, speech_lens, 6)
            mask_expanded = encoder_out_mask.squeeze(1).unsqueeze(-1).expand(
                -1, -1, encoder_out.size(-1))
            selected_samples = encoder_out[mask_expanded].view(
                -1, encoder_out.size(-1))
            info_dict = kmeans_model(selected_samples)
            if rank == 0 and i % 100 == 0:
                print(
                    "iter: {} | rank {} |  inertias: {} |\n means: {} ".format(
                        i, rank, info_dict['inertias'], info_dict['means']))
                if i % args.save_interval == 0:
                    torch.save(
                        kmeans_model.state_dict(),
                        os.path.join(args.kmeans_model_dir,
                                     "km_iter_{}.pt".format(i)))

        dist.barrier()

    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        torch.save(kmeans_model.state_dict(),
                   os.path.join(args.kmeans_model_dir, "km_final.pt"))


if __name__ == '__main__':
    main()
