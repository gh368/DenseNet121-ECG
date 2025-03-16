__all__ = ('BasicBlock', 'DenseNet1D') 

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1, expansion=None):
        super(BottleneckBlock, self).__init__()
        if out_chan // in_chan == 4:  # first block
            mid_chan = in_chan
        elif out_chan // in_chan == 2:  # first in block sequence
            mid_chan = in_chan // 2
        elif out_chan == in_chan:  # not first in block sequence
            mid_chan = in_chan // 4

        self.conv1 = nn.Conv1d(in_chan, mid_chan, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_chan)
        self.conv2 = nn.Conv1d(mid_chan, mid_chan, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_chan)
        self.conv3 = nn.Conv1d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_chan != out_chan:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_chan)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    """
    Kept for compatibility in case other parts of the code rely on 'BasicBlock'.
    """
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, expansion=None):
        super().__init__()
        if stride > 1 or in_chan != out_chan:
            self.shortcut = conv1d(in_chan, out_chan, 1, stride)
        else:
            self.shortcut = nn.Identity()
        self.bn1 = nn.BatchNorm1d(in_chan)
        self.relu1 = nn.ReLU()
        self.conv1 = conv1d(in_chan, out_chan, kernel_size, stride)
        self.bn2 = nn.BatchNorm1d(out_chan)
        self.relu2 = nn.ReLU()
        self.conv2 = conv1d(out_chan, out_chan, kernel_size)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        out = x + shortcut
        return out


########################################
# DenseNet Components for 1D Signals
########################################
class _DenseLayer1D(nn.Module):
    """
    A single layer within a 1D Dense Block:
      - BN -> ReLU -> Conv -> optional Dropout
      - Output is concatenated with the input
    """
    def __init__(self, in_channels, growth_rate, kernel_size=3, drop_rate=0.0):
        super(_DenseLayer1D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels,
            growth_rate,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class _DenseBlock1D(nn.Module):
    """
    A block consisting of multiple dense layers in sequence.
    """
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super(_DenseBlock1D, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                _DenseLayer1D(
                    in_channels + i * growth_rate,
                    growth_rate,
                    kernel_size=3,
                    drop_rate=drop_rate
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition1D(nn.Module):
    """
    Transition layer between dense blocks:
      - BN -> ReLU -> 1D Conv -> AvgPool(stride=2)
    Reduces both feature maps and sequence length.
    """
    def __init__(self, in_channels, out_channels):
        super(_Transition1D, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


#########################################
# 1D DenseNet-121 Implementation
#########################################
class DenseNet1D(nn.Module):
    """
    A 1D DenseNet-121 model. Uses block_config=(6,12,24,16) by default.
    """
    def __init__(
        self, stages, num_outputs,
        in_chan=1,
        out_chan=64,         # initial output channels from the first conv
        kernel_size=None,    # retained for structure, not directly used
        stem_kernel_size=7,  # front conv kernel size
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        drop_rate=0.0
    ):
        super().__init__()

        # 1) Initial convolution + BN + ReLU + MaxPool
        padding = (stem_kernel_size - 1) // 2
        self.conv0 = nn.Conv1d(
            in_chan,
            out_chan,
            kernel_size=stem_kernel_size,
            stride=2,
            padding=padding,
            bias=False
        )
        self.bn0 = nn.BatchNorm1d(out_chan)
        self.relu0 = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 2) Dense Blocks + Transitions
        num_features = out_chan
        self.denseblocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            denseblock = _DenseBlock1D(num_layers, num_features, growth_rate, drop_rate)
            self.denseblocks.append(denseblock)
            num_features = num_features + num_layers * growth_rate

            # If not the last block, add a transition
            if i != len(block_config) - 1:
                out_features = num_features // 2
                trans = _Transition1D(num_features, out_features)
                self.transitions.append(trans)
                num_features = out_features

        # 3) Final BN + ReLU
        self.bn_final = nn.BatchNorm1d(num_features)
        self.relu_final = nn.ReLU(inplace=True)

        # 4) Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_features, num_outputs)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _init_stage(block, num_blocks, in_chan, out_chan, kernel_size, stride):
        """
        Retained for structural compatibility, but unused by DenseNet1D.
        """
        raise NotImplementedError("_init_stage is not used in DenseNet1D")

    def forward(self, x):
        # Initial layers
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        # Dense Blocks + Transitions
        for i, denseblock in enumerate(self.denseblocks):
            x = denseblock(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # Final BN + ReLU
        x = self.bn_final(x)
        x = self.relu_final(x)

        # Global average pool
        x = self.avgpool(x).squeeze(-1)

        # Classification head
        out = self.fc(x)
        return out

###############################################################
def densenet1d_121(num_outputs, **kwargs):
    """
    Returns a 1D DenseNet-121 (block_config=(6,12,24,16)), ignoring 'stages'.
    """
    return DenseNet1D(
        stages=[2, 2, 2, 2],  # Ignored for DenseNet, but kept for structure
        num_outputs=num_outputs,
        block_config=(6, 12, 24, 16),
        **kwargs
    )


def densenet1d_121_alt(num_outputs, **kwargs):
    """
    Alternative 1D DenseNet-121 variant (same block_config, different name).
    """
    return DenseNet1D(
        stages=[3, 4, 6, 3],  # Ignored for DenseNet, but kept for structure
        num_outputs=num_outputs,
        block_config=(6, 12, 24, 16),
        **kwargs
    )


###############################################
# conv1d helper (unchanged, simply retained)
###############################################
def conv1d(in_chan, out_chan, kernel_size, stride=1):
    assert kernel_size % 2 == 1, "Kernel size must be odd."
    padding = (kernel_size - 1) // 2
    return nn.Conv1d(
        in_chan,
        out_chan,
        kernel_size,
        stride,
        padding,
        bias=True
    )
