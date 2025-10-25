from collections.abc import Iterable
import itertools

import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import nn

class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return x

class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.prompt_param = nn.Parameter(torch.rand(1, nof_kernels, hidden_dim))
        self.to_scores = nn.Sequential(nn.Linear(c_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       )

    def forward(self, x, temperature=1):
        x = self.global_pooling(x)
        out = self.to_scores(x).unsqueeze(1)
        scores = torch.mul(out, self.prompt_param).sum(-1)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(TempModule):
    def __init__(self, nof_kernels, extend, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, temperature=1,  bias=True):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: size of the kernel.
        :param groups: controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups
        self.temperature = temperature
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(in_channels, max(1, in_channels * extend), nof_kernels)
        self.kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True)
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x):
        batch_size = x.shape[0]

        alphas = self.attention(x, self.temperature)
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])  # batch_size*out_c X in_c X kernel_size X kernel_size
        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])  # 1 X batch_size*out_c X H X W

        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size,
                       **self.conv_args)  # 1 X batch_size*out_C X H' x W'
        out = out.view(batch_size, -1, *out.shape[-2:])  # batch_size X out_C X H' x W'

        return out


class FlexibleKernelsDynamicConvolution:
    def __init__(self, Base, nof_kernels, reduce):
        if isinstance(nof_kernels, Iterable):
            self.nof_kernels_it = iter(nof_kernels)
        else:
            self.nof_kernels_it = itertools.cycle([nof_kernels])
        self.Base = Base
        self.reduce = reduce

    def __call__(self, *args, **kwargs):
        return self.Base(next(self.nof_kernels_it), self.reduce, *args, **kwargs)


def dynamic_convolution_generator(nof_kernels, reduce):
    return FlexibleKernelsDynamicConvolution(DynamicConvolution, nof_kernels, reduce)


if __name__ == '__main__':
    torch.manual_seed(41)
    t = torch.randn(5, 64, 128, 128)
    conv = DynamicConvolution(16, 1, in_channels=64, out_channels=64, kernel_size=3, padding=1, temperature=1, bias=True)
    print(conv(t).sum())
