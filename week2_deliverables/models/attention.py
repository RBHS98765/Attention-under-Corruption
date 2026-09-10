"""BAM and CBAM attention blocks, adapted from the Week 1 verified upstream code.

Source repository: https://github.com/Jongchan/attention-module
Commit: 459efad0e05ee7dde50c41ca10a3d0800bc3792a (MIT license)

BAM patch:
  The upstream BAM constructor contains one dead, undefined assignment
  ``self.gate_activation = gate_activation`` in ``ChannelGate.__init__`` that
  raises ``NameError: name 'gate_activation' is not defined`` at construction.
  The Week 1 verification confirmed this is a genuine upstream bug. The project
  copy below removes ONLY that single dead line (the same change saved as
  ``bam_gate_activation_fix.patch``). The forward computation is otherwise
  byte-for-byte identical to the verified upstream code. The upstream
  repository itself is never modified; this file is the project working copy.

Forward semantics (unchanged from upstream):
  - BAM:  att = 1 + sigmoid(channel_att(x) * spatial_att(x));  out = att * x
  - CBAM: channel gate (avg+max pooled MLP, sigmoid) then spatial gate
          (channel-pooled 7x7 conv, sigmoid).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """No-op attention used for the plain CNN baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# BAM (Bottleneck Attention Module) - project copy, one-line patch applied
# ---------------------------------------------------------------------------
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BAMChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(BAMChannelGate, self).__init__()
        # NOTE: the upstream dead line "self.gate_activation = gate_activation"
        # is removed here (see module docstring and bam_gate_activation_fix.patch).
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module('gate_c_fc_%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1), nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())
        self.gate_c.add_module('gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class BAMSpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(BAMSpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module('gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1))
        self.gate_s.add_module('gate_s_bn_reduce0', nn.BatchNorm2d(gate_channel // reduction_ratio))
        self.gate_s.add_module('gate_s_relu_reduce0', nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module('gate_s_conv_di_%d' % i, nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio, kernel_size=3, padding=dilation_val, dilation=dilation_val))
            self.gate_s.add_module('gate_s_bn_di_%d' % i, nn.BatchNorm2d(gate_channel // reduction_ratio))
            self.gate_s.add_module('gate_s_relu_di_%d' % i, nn.ReLU())
        self.gate_s.add_module('gate_s_conv_final', nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1))

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    """Bottleneck Attention Module.

    Upstream signature extended with optional reduction/dilation settings so
    the configuration freeze can record them; defaults match upstream exactly
    (reduction_ratio=16, dilation_conv_num=2, dilation_val=4). Forward is
    unchanged from the verified upstream code.
    """

    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(BAM, self).__init__()
        self.gate_channel = gate_channel
        self.reduction_ratio = reduction_ratio
        self.dilation_conv_num = dilation_conv_num
        self.dilation_val = dilation_val
        self.channel_att = BAMChannelGate(gate_channel, reduction_ratio, num_layers=1)
        self.spatial_att = BAMSpatialGate(gate_channel, reduction_ratio, dilation_conv_num, dilation_val)

    def forward(self, in_tensor):
        att = 1 + F.sigmoid(self.channel_att(in_tensor) * self.spatial_att(in_tensor))
        return att * in_tensor


# ---------------------------------------------------------------------------
# CBAM (Convolutional Block Attention Module) - project copy, pristine
# ---------------------------------------------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CBAMChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(CBAMChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError(f"unsupported CBAM pool_type: {pool_type}")
            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class CBAMSpatialGate(nn.Module):
    def __init__(self):
        super(CBAMSpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False):
        super(CBAM, self).__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.ChannelGate = CBAMChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = CBAMSpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
