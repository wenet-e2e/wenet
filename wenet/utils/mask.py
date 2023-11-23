# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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


from typing import Optional, Union

import torch

'''
def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
'''

def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)

def causal_or_lookahead_mask(
    mask: torch.Tensor,
    right_context: int,
    left_context: int,
    left_t_valid: int = 0,
) -> torch.Tensor:
    """Create mask (B, T, T) with history or future or both,
       this is for causal or noncausal streaming encoder

    Args:
        mask (torch.Tensor): size of mask shape (B, 1, T)
        right_context (int): future context size
        left_context (int): history context size
        left_t_valid (int): valid start offset

    Returns:
        torch.Tensor: mask shape (B, T, T)

    Examples:
        >>> seq_len  = torch.tensor([2,3,4])
        >>> seq_mask = make_non_pad_mask(seq_len)
        [[1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 0, 2)
        [[[1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 1, 1, 1]]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 1, 2)
        [[[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1],
         [0, 1, 1, 1]]]
    """
    _, _, T = mask.size()
    indices = torch.arange(T, device=mask.device)
    start = torch.where(indices > left_context, indices - left_context, 0)
    start = torch.where(indices < left_t_valid, indices, start).unsqueeze(1)

    end = indices + right_context + 1
    end = end.unsqueeze(1)
    indices_expand = indices.unsqueeze(0)
    gt = (indices_expand >= start)
    lt = (indices_expand < end)

    return (gt & lt) * mask.transpose(1, 2) * mask


def t_sequence_mask(batch_size: int,
                    choose_range: torch.Tensor,
                    mask_size: int,
                    max_length: Optional[Union[torch.Tensor, int]] = None,
                    masks_per_frame: float = 0.0,
                    max_ratio: float = 1.0,
                    dtype=torch.float32,
                    multiplicity: int = 1,
                    fix_length: bool = False,
                    device: torch.device = torch.device('cpu')):
    """Create random mask (B, T) with spans

    Args:
        batch_size (int): batch size
        choose_range (torch.Tensor): range within which the masked entries must
            lie. shape (batch_size,).
        mask_size (int):  size of the mask.
        max_length (torch.Tensor|None):  maximum number of allowed consecutive
            masked entries.
        masks_per_frame (float): number of masks per frame. If > 0, the
            multiplicity of the mask is set to be masks_per_frame * choose_range
            If masks_per_frame == 0, all the masks are composed.The masked
            regions are set to zero.
        max_ratio (float): maximum portion of the entire range allowed to be
            masked.
        multiplicity (int): maximum number of total masks.
        fix_length (bool): if spans is fix length or < rand(max_length)

    Returns:
        torch.Tensor: mask shape (B, T)

    """

    if max_length is not None:
        assert max_length > 0
        if isinstance(max_length, int):
            max_length = torch.tensor(max_length, device=device)
        max_length = torch.broadcast_to(max_length.to(dtype), (batch_size, ))
    else:
        max_length = choose_range.to(dtype) * max_ratio

    masked_portion = torch.rand(batch_size, multiplicity, dtype=dtype)
    masked_frame_size = max_length.unsqueeze(1) * masked_portion
    masked_frame_size = masked_frame_size.to(torch.int32)

    # Make sure the sampled length was sampler than max_ratio * length_bound.
    choose_range = choose_range.unsqueeze(-1)
    choose_range = torch.tile(choose_range, [1, multiplicity])
    length_bound = choose_range.to(dtype) * max_ratio
    length_bound = length_bound.to(torch.int32)
    length = torch.minimum(
        masked_frame_size,
        torch.maximum(length_bound, torch.tensor(1,
                                                 device=length_bound.device)))
    if fix_length:
        length = max_length.unsqueeze(1).repeat(1, multiplicity) - 1

    # Choose random starting point
    random_start = torch.rand(batch_size, multiplicity)
    start_with_in_valid_range = random_start * (choose_range - length + 1)
    start = start_with_in_valid_range.to(torch.int32)
    end = start + length - 1

    # Shift starting and end point by small value
    delta = 0.1
    start = (start.to(dtype) - delta).unsqueeze(-1)
    start = torch.tile(start, [1, 1, mask_size])
    end = (end.to(dtype) + delta).unsqueeze(-1)
    end = torch.tile(end, [1, 1, mask_size])

    # Construct pre-mask of size (batch_size, multiplicity, mask_size)
    diagonal = torch.arange(end=mask_size,
                            dtype=dtype).unsqueeze(0).unsqueeze(0)
    diagonal = torch.tile(diagonal, [batch_size, multiplicity, 1])
    pre_mask = torch.logical_and(diagonal < end, diagonal > start).to(dtype)

    # Sum masks with ppropriate multiplicity.
    if masks_per_frame > 0:
        multiplicity_weights = torch.arange(end=multiplicity,
                                            dtype=torch.int32).to(dtype)
        multiplicity_weights = multiplicity_weights.unsqueeze(0)
        multiplicity_weights = torch.tile(multiplicity_weights,
                                          [batch_size, 1])
        multiplicity_tensor = masks_per_frame * choose_range.to(dtype)
        multiplicity_weights = (multiplicity_weights <
                                multiplicity_tensor).to(dtype)
        pre_mask = (pre_mask * multiplicity_weights.unsqueeze(1)).sum(
            1)  # [B,T]
    else:
        pre_mask = pre_mask.sum(1)  # [B,T]
    mask = (1.0 - (pre_mask > 0).to(dtype).to(dtype))

    return mask


def time_mask(inputs: torch.Tensor,
              inputs_len: torch.Tensor,
              num_t_mask: int = 2,
              max_t: int = 50):

    if num_t_mask <= 0:
        return inputs
    B, T, _ = inputs.size()
    time_mask = t_sequence_mask(
        B,
        inputs_len,
        T,
        max_t,
        0.0,
        multiplicity=num_t_mask,
        device=inputs.device,
    )
    return inputs * time_mask.unsqueeze(2)


def freq_mask(inputs: torch.Tensor, num_f_mask=2, max_f=10):
    if num_f_mask <= 0:
        return inputs
    B, _, F = inputs.size()
    f = torch.tensor(F, device=inputs.device)
    choose_range = torch.broadcast_to(f, (B, )).to(torch.int32)
    freq_mask = t_sequence_mask(
        B,
        choose_range,
        F,
        max_f,
        0.0,
        multiplicity=num_f_mask,
        device=inputs.device,
    )
    return inputs * freq_mask.unsqueeze(1)
