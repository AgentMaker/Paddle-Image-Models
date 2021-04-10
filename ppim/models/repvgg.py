import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ..units import load_model


transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'repvgg_a0': r'https://bj.bcebos.com/v1/ai-studio-online/26d1d26e0d0141deafeb7e9980ec8b5a555232b938e44fefa93da930422af42b?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A0.pdparams',
    'repvgg_a1': r'https://bj.bcebos.com/v1/ai-studio-online/afa4629fb917427a829bb278250b84b0380d580b40fc4e478eb5fdb75fe22096?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A1.pdparams',
    'repvgg_a2': r'https://bj.bcebos.com/v1/ai-studio-online/200f4d6038834fd49796941f5acf65308e6e096d2b8c496abb9d1c0204f44cb1?responseContentDisposition=attachment%3B%20filename%3DRepVGG_A2.pdparams',
    'repvgg_b0': r'https://bj.bcebos.com/v1/ai-studio-online/93c345b4a76b4f88b3590fa703a270b009cc9c05481640a49e8654222459e79f?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B0.pdparams',
    'repvgg_b1': r'https://bj.bcebos.com/v1/ai-studio-online/b2f8171754bd4d3cb44739b675dc1f0b8cb77ebefdad47ec82ce98292726bf2c?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1.pdparams',
    'repvgg_b2': r'https://bj.bcebos.com/v1/ai-studio-online/9fc65aab46b441dca194f974bdf420710b2144e941704330869d62a2ab9cb0b6?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B2.pdparams',
    'repvgg_b3': r'https://bj.bcebos.com/v1/ai-studio-online/8d902ba9ebf3441e896e8d7078544005a0715ca6867f4067989dcc533ace2435?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B3_200epochs.pdparams',
    'repvgg_b1g2': r'https://bj.bcebos.com/v1/ai-studio-online/da4931eff12142a290ce8d01a0cd3b777a81b53c971b4dd2a1a627c615466570?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1g2.pdparams',
    'repvgg_b1g4': r'https://bj.bcebos.com/v1/ai-studio-online/440040d200b14bcb9951e47877b7b416454affd75f8e4eaba6fedfa87c4ab66a?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B1g4.pdparams',
    'repvgg_b2g4': r'https://bj.bcebos.com/v1/ai-studio-online/42b0654c15f942c9828a7ca7d117638417c48ccdeac84123bcd72558db7a01c2?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B2g4_200epochs.pdparams',
    'repvgg_b3g4': r'https://bj.bcebos.com/v1/ai-studio-online/5e4f6084ee954a319c2e0c11aadae680c643ae88bdbb44d2a1875a38f5278060?responseContentDisposition=attachment%3B%20filename%3DRepVGG_B3g4_200epochs.pdparams'
}


class ConvBN(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias_attr=False)
        self.bn = nn.BatchNorm2D(num_features=out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y


class RepVGGBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.rbr_identity = nn.BatchNorm2D(
            num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = ConvBN(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = ConvBN(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if not self.training:
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def eval(self):
        if not hasattr(self, 'rbr_reparam'):
            self.rbr_reparam = nn.Conv2D(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                         padding=self.padding, dilation=self.dilation, groups=self.groups, padding_mode=self.padding_mode)
        self.training = False
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        for layer in self.sublayers():
            layer.eval()

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvBN):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class RepVGG(nn.Layer):
    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, class_dim=1000, with_pool=True):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4

        self.class_dim = class_dim
        self.with_pool = with_pool
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        if with_pool:
            self.gap = nn.AdaptiveAvgPool2D(output_size=1)

        if class_dim > 0:
            self.linear = nn.Linear(int(512 * width_multiplier[3]), class_dim)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        if self.with_pool:
            out = self.gap(out)

        if self.class_dim > 0:
            out = paddle.flatten(out, start_axis=1)
            out = self.linear(out)

        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def repvgg_a0(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_a0'])
    return model, transforms


def repvgg_a1(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_a1'])
    return model, transforms


def repvgg_a2(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_a2'])
    return model, transforms


def repvgg_b0(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b0'])
    return model, transforms


def repvgg_b1(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b1'])
    return model, transforms


def repvgg_b2(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b2'])
    return model, transforms


def repvgg_b3(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b3'])
    return model, transforms


def repvgg_b1g2(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b1g2'])
    return model, transforms


def repvgg_b1g4(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b1g4'])
    return model, transforms


def repvgg_b2g4(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b2g4'])
    return model, transforms


def repvgg_b3g4(pretrained=False, **kwargs):
    model = RepVGG(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['repvgg_b3g4'])
    return model, transforms
