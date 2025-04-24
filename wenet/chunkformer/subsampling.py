"""Subsampling layer definition."""


import torch
import math
from wenet.utils.mask import make_pad_mask

class DepthwiseConvSubsampling(torch.nn.Module):
    """
    Args:
        subsampling (str): The subsampling technique
        subsampling_rate (int): The subsampling factor which should be a power of 2
        subsampling_conv_chunking_factor (int): Input chunking factor
        1 (auto) or a power of 2. Default is 1
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self,
        subsampling,
        subsampling_rate,
        feat_in,
        feat_out,
        conv_channels,
        pos_enc_class: torch.nn.Module,
        subsampling_conv_chunking_factor=1,
        activation=torch.nn.ReLU(),
    ):
        super(DepthwiseConvSubsampling, self).__init__()
        self._subsampling = subsampling
        self._conv_channels = conv_channels
        self._feat_in = feat_in
        self._feat_out = feat_out
        self.pos_enc = pos_enc_class

        if subsampling_rate % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_rate, 2))
        self.subsampling_rate = subsampling_rate
        self.right_context = 14

        if (
            subsampling_conv_chunking_factor != -1
            and subsampling_conv_chunking_factor != 1
            and subsampling_conv_chunking_factor % 2 != 0
        ):
            raise ValueError("""subsampling_conv_chunking_factor
                                "should be -1, 1, or a power of 2""")
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        in_channels = 1
        layers = []

        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False


        self._left_padding = 0
        self._right_padding = 0
        self._max_cache_len = 0

        # Layer 1
        layers.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=0,
            )
        )
        in_channels = conv_channels
        layers.append(activation)

        for _ in range(self._sampling_num - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    padding=0,
                    groups=in_channels,
                )
            )

            layers.append(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = self.calc_length(
            lengths=in_length
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv2d_subsampling = True


        self.conv = torch.nn.Sequential(*layers)

    def get_sampling_frames(self):
        return [1, self.subsampling_rate]

    def get_streaming_cache_size(self):
        return [0, self.subsampling_rate + 1]

    def forward(self,
                x,
                mask,
                chunk_size: int = -1,
                left_context_size: int = 0,
                right_context_size: int = 0):
        lengths = mask.sum(dim=-1).squeeze(-1)
        lengths = self.calc_length(
            lengths,
        )

        # Unsqueeze Channel Axis
        if self.conv2d_subsampling:
            x = x.unsqueeze(1)
        # Transpose to Channel First mode
        else:
            x = x.transpose(1, 2)

        # split inputs if chunking_factor is set
        if self.subsampling_conv_chunking_factor != -1 and self.conv2d_subsampling:
            if self.subsampling_conv_chunking_factor == 1:
                # if subsampling_conv_chunking_factor is 1, we split only if needed
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
                if torch.numel(x) > x_ceil:
                    need_to_split = True
                else:
                    need_to_split = False
            else:
                # if subsampling_conv_chunking_factor > 1 we always split
                need_to_split = True

            # need_to_split = False
            if need_to_split:
                x, success = self.conv_split_by_batch(x)
                # success = False
                if not success:  # if unable to split by batch, try by channel
                    x = self.conv_split_by_channel(x)
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        # Flatten Channel and Frequency Axes
        if self.conv2d_subsampling:
            b, c, t, f = x.size()
            x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        # Transpose to Channel Last mode
        else:
            x = x.transpose(1, 2)
        x, pos_emb = self.pos_enc(
            x, 
            chunk_size=chunk_size, 
            left_context_size=left_context_size,
            right_context_size=right_context_size)
        mask = ~make_pad_mask(lengths, x.size(1)).unsqueeze(1)
        return x, pos_emb, mask

    def reset_parameters(self):
        # initialize weights
        with torch.no_grad():
            # init conv
            scale = 1.0 / self._kernel_size
            dw_max = (self._kernel_size ** 2) ** -0.5
            pw_max = self._conv_channels ** -0.5

            torch.nn.init.uniform_(self.conv[0].weight, -scale, scale)
            torch.nn.init.uniform_(self.conv[0].bias, -scale, scale)

            for idx in range(2, len(self.conv), 3):
                torch.nn.init.uniform_(self.conv[idx].weight, -dw_max, dw_max)
                torch.nn.init.uniform_(self.conv[idx].bias, -dw_max, dw_max)
                torch.nn.init.uniform_(self.conv[idx + 1].weight, -pw_max, pw_max)
                torch.nn.init.uniform_(self.conv[idx + 1].bias, -pw_max, pw_max)

            fc_scale = (self._feat_out * self._feat_in / self._sampling_num) ** -0.5
            torch.nn.init.uniform_(self.out.weight, -fc_scale, fc_scale)
            torch.nn.init.uniform_(self.out.bias, -fc_scale, fc_scale)

    def conv_split_by_batch(self, x):
        """ Tries to split input by batch, run conv and concat results """
        b, _, _, _ = x.size()
        if b == 1:  # can't split if batch size is 1
            return x, False

        if self.subsampling_conv_chunking_factor > 1:
            cf = self.subsampling_conv_chunking_factor
        else:
            # avoiding a bug / feature limiting indexing of tensors to 2**31
            # see https://github.com/pytorch/pytorch/issues/80020
            x_ceil = 2 ** 31 / self._conv_channels * self._stride * self._stride
            p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
            cf = 2 ** p

        new_batch_size = b // cf
        if new_batch_size == 0:  # input is too big
            return x, False

        return torch.cat([self.conv(chunk)
                          for chunk in torch.split(x, new_batch_size, 0)]), True

    def conv_split_by_channel(self, x):
        """ For dw convs, tries to split input by time, run conv and concat results """
        x = self.conv[0](x)  # full conv2D
        x = self.conv[1](x)  # activation

        for i in range(self._sampling_num - 1):
            _, c, t, _ = x.size()

            if self.subsampling_conv_chunking_factor > 1:
                cf = self.subsampling_conv_chunking_factor
            else:
                # avoiding a bug / feature limiting indexing of tensors to 2**31
                # see https://github.com/pytorch/pytorch/issues/80020
                p = math.ceil(math.log(torch.numel(x) / 2 ** 31, 2))
                cf = 2 ** p

            new_c = int(c // cf)
            if new_c == 0:
                new_c = 1

            new_t = int(t // cf)
            if new_t == 0:
                new_t = 1

            x = self.channel_chunked_conv(self.conv[i * 3 + 2], new_c, x)

            # splitting pointwise convs by time
            x = torch.cat([self.conv[i * 3 + 3](chunk)
                           for chunk in torch.split(x, new_t, 2)], 2)
            x = self.conv[i * 3 + 4](x)  # activation
        return x

    def channel_chunked_conv(self, conv, chunk_size, x):
        """ Performs channel chunked convolution"""

        ind = 0
        out_chunks = []
        for chunk in torch.split(x, chunk_size, 1):
            step = chunk.size()[1]
            ch_out = torch.nn.functional.conv2d(
                chunk,
                conv.weight[ind : ind + step, :, :, :],
                bias=conv.bias[ind : ind + step],
                stride=self._stride,
                padding=self._left_padding,
                groups=step,
            )
            out_chunks.append(ch_out)
            ind += step

        return torch.cat(out_chunks, 1)

    def calc_length(self, lengths):
        """
        Calculates the output length of a Tensor
        passed through a convolution or max pooling layer
        """
        all_paddings = self._left_padding + self._right_padding
        kernel_size = self._kernel_size
        stride = self._stride
        ceil_mode = self._ceil_mode
        repeat_num = self._sampling_num
        add_pad = all_paddings - kernel_size
        one = 1.0
        for i in range(repeat_num):
            lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
            if ceil_mode:
                lengths = torch.ceil(lengths)
            else:
                lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)
