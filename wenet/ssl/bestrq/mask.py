import torch
import numpy as np


def _sampler(pdf: torch.Tensor, num_samples: int,
             device=torch.device('cpu')) -> torch.Tensor:
    size = pdf.size()
    z = -torch.log(torch.rand(size, device=device))
    _, indices = torch.topk(pdf + z, num_samples)
    return indices


def compute_mask_indices(
        size: torch.Size,
        mask_prob: float,
        mask_length: int,
        min_masks: int = 0,
        device=torch.device('cpu'),
) -> torch.Tensor:

    assert len(size) == 2
    batch_size, seq_length = size

    # compute number of masked span in batch
    num_masked_spans = mask_prob * float(seq_length) / float(
        mask_length) + torch.rand(1)[0]
    num_masked_spans = int(num_masked_spans)
    num_masked_spans = max(num_masked_spans, min_masks)

    # num_masked <= seq_length
    if num_masked_spans * mask_length > seq_length:
        num_masked_spans = seq_length // mask_length

    pdf = torch.ones(batch_size, seq_length - (mask_length - 1), device=device)
    mask_idxs = _sampler(pdf, num_masked_spans, device=device)

    mask_idxs = mask_idxs.unsqueeze(-1).repeat(1, 1, mask_length).view(
        batch_size,
        num_masked_spans * mask_length)  # [B,num_masked_spans*mask_length]

    offset = torch.arange(mask_length, device=device).view(1, 1, -1).repeat(
        1, num_masked_spans, 1)  # [1,num_masked_spans,mask_length]
    offset = offset.view(1, num_masked_spans * mask_length)

    mask_idxs = mask_idxs + offset  # [B,num_masked_spans, mask_length]

    ones = torch.ones(batch_size,
                      seq_length,
                      dtype=torch.bool,
                      device=mask_idxs.device)
    # masks to fill
    full_mask = torch.zeros_like(ones,
                                 dtype=torch.bool,
                                 device=mask_idxs.device)
    return torch.scatter(full_mask, dim=1, index=mask_idxs, src=ones)


def compute_mask_indices_v2(
        shape,
        padding_mask,
        mask_prob: float,
        mask_length: int,
        mask_type: str = 'static',
        mask_other: float = 0.0,
        min_masks: int = 2,
        no_overlap: bool = False,
        min_space: int = 1,
        device=torch.device('cpu'),
):
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    padding_mask = padding_mask.cpu().numpy()
    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length) + np.random.rand())

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None and not isinstance(padding_mask, bytes):
            sz = all_sz - padding_mask[i].sum()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == 'static':
            lengths = np.full(num_mask, mask_length)
        elif mask_type == 'uniform':
            lengths = np.random.randint(mask_other,
                                        mask_length * 2 + 1,
                                        size=num_mask)
        elif mask_type == 'normal':
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == 'poisson':
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception('unknown mask selection ' + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length, mask_idc):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0
                     for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length, mask_idc))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray([
                mask_idc[j] + offset for j in range(len(mask_idc))
                for offset in range(lengths[j])
            ])

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    mask = torch.from_numpy(mask).to(device)
    return mask
