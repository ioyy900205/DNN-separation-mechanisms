import torch
import torch.nn as nn 
import torch.nn.functional as F
_NORM='BN'
def get_norm(name,channel):
    return nn.BatchNorm2d(channel)

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x

class _OSA_module(nn.Module):
    def __init__(
        self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=True, identity=True, depthwise=True
    ):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.depthwise = depthwise
        layers = []
        self.isConcat = (in_ch != stage_ch or depthwise)

        # Layer reduction for depthwise convolution
        reduce = depthwise and in_ch != stage_ch
        if reduce:
            layers.append(self._conv1x1_layer(in_ch, stage_ch, module_name + "_reduction", "0"))

        # Convolutional layers
        for i in range(layer_per_block):
            conv_layer = self._dw_conv3x3_layer if depthwise else self._conv3x3_layer
            layers.append(conv_layer(stage_ch, module_name, str(i)))

        self.layers = nn.ModuleList(layers)

        # Feature concatenation
        concat_in_ch = in_ch + layer_per_block * stage_ch if self.isConcat else layer_per_block * stage_ch
        if reduce:
            concat_in_ch += stage_ch

        self.concat_conv = self._conv1x1_layer(concat_in_ch, concat_ch, module_name, "concat")

        # Squeeze and Excitation layer
        self.ese = eSEModule(concat_ch) if SE else nn.Identity()

    def forward(self, x):
        identity_feat = x

        output = [x] if self.isConcat else []
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)

        x = self.concat_conv(x)
        x = self.ese(x)

        if self.identity:
            x = x + identity_feat

        return x

    @staticmethod
    def _conv3x3_layer(in_channels, module_name, postfix, stride=1, padding=1):
        """Creates a standard 3x3 convolution layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding, bias=False),
            get_norm(_NORM, in_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _dw_conv3x3_layer(out_channels, module_name, postfix, stride=1, padding=1):
        """Creates a depthwise 3x3 convolution layer followed by 1x1 pointwise conv"""
        return nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride, padding, groups=out_channels, bias=False),
            get_norm(_NORM, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            get_norm(_NORM, out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _conv1x1_layer(in_channels, out_channels, module_name, postfix, stride=1, padding=0):
        """Creates a 1x1 convolution layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=False),
            get_norm(_NORM, out_channels),
            nn.ReLU(inplace=True),
        )


if __name__ == "__main__":
    net = _OSA_module(in_ch=10,stage_ch=20,concat_ch=10,layer_per_block=4,module_name='osa_moduel',depthwise=True)
    input = torch.randn(2,10,40,40)
    output = net(input)