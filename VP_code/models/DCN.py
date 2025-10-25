import torchvision
import math
from torch.nn.modules.utils import _pair, _single
import torch
import torch.nn as nn
class ModulatedDeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    # def forward(self, x, offset, mask):
    #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation,
    #                                  self.groups, self.deformable_groups)
class ModulatedDeformConvPack(ModulatedDeformConv):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(ModulatedDeformConvPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_weights()

    def init_weights(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()
class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """Flow-guided deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
        pa_frames (int): The number of parallel warping frames. Default: 2.

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super(DCNv2PackFlowGuided, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d((1+self.pa_frames//2) * self.in_channels + self.pa_frames, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 2 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        super(ModulatedDeformConvPack, self).init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x, x_flow_warpeds, x_current, flows):
        out = self.conv_offset(torch.cat([x_flow_warpeds, x_current, flows], dim=1))
        o1, o2 = torch.chunk(out, 2, dim=1) # c 3*3 h, w

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1)//2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # mask
        # mask = torch.sigmoid(mask)
        mask = None

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, mask)

if __name__ == '__main__':
    DD = DCNv2PackFlowGuided(16, 16, kernel_size=3, stride=1, padding=1, deformable_groups=1, pa_frames=2)
    x = torch.randn(1, 16, 64, 64)
    f = torch.randn(1, 1, 2, 64, 64)
    y = DD(x, x, x, f[:,0,:,:])