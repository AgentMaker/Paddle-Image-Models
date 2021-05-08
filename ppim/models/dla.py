import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T

from paddle.nn.initializer import Normal

from ppim.models.common import Identity
from ppim.models.common import load_model
from ppim.models.common import zeros_, ones_


transforms = T.Compose([
    T.Resize(256, interpolation='bilinear'),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'dla_34': r'https://bj.bcebos.com/v1/ai-studio-online/a4e08c790f0247c8ab44cfa9ec6264720a3fab64b51d4ee88d0e7d3511e6348a?responseContentDisposition=attachment%3B%20filename%3Ddla34%2Btricks.pdparams',
    'dla_46_c': r'https://bj.bcebos.com/v1/ai-studio-online/245e16ae6b284b368798a6f8e3cf068e55eea96e22724ec5bff8d146c64da990?responseContentDisposition=attachment%3B%20filename%3Ddla46_c.pdparams',
    'dla_46x_c': r'https://bj.bcebos.com/v1/ai-studio-online/b295201d245247fb8cd601b60919cabf5df51a8997d04380bd07eac71e4152dd?responseContentDisposition=attachment%3B%20filename%3Ddla46x_c.pdparams',
    'dla_60': r'https://bj.bcebos.com/v1/ai-studio-online/e545d431a9f84bb4aecd2c75e34e6169503be2d2e8d246cb9cff393559409f7b?responseContentDisposition=attachment%3B%20filename%3Ddla60.pdparams',
    'dla_60x': r'https://bj.bcebos.com/v1/ai-studio-online/a07ea1cec75a460ebf6dcace4ab0c8c28e923af88dd74573baaaa6db8738168d?responseContentDisposition=attachment%3B%20filename%3Ddla60x.pdparams',
    'dla_60x_c': r'https://bj.bcebos.com/v1/ai-studio-online/0c15f589fa524d1dbe753afe2619f2fe33773c0ca6db4966a3ab8f755fca3c98?responseContentDisposition=attachment%3B%20filename%3Ddla60x_c.pdparams',
    'dla_102': r'https://bj.bcebos.com/v1/ai-studio-online/288ca91946d04df891750eed67b3070ec38a29e9a7b24eff90c0e397d3b82c7f?responseContentDisposition=attachment%3B%20filename%3Ddla102%2Btricks.pdparams',
    'dla_102x': r'https://bj.bcebos.com/v1/ai-studio-online/0653e6aae7594e2a8de94728f6656c375557f7960a8949a1926eb017e978c477?responseContentDisposition=attachment%3B%20filename%3Ddla102x.pdparams',
    'dla_102x2': r'https://bj.bcebos.com/v1/ai-studio-online/80cd37d877974ad18d1ccefdae2a5c2cce1cba2831544deeaea1fa672343cc17?responseContentDisposition=attachment%3B%20filename%3Ddla102x2.pdparams',
    'dla_169': r'https://bj.bcebos.com/v1/ai-studio-online/f299fab9020344d4aee7ccf3a79e98858494e0536bca4703a5f5152747395cca?responseContentDisposition=attachment%3B%20filename%3Ddla169.pdparams'
}


class DlaBasic(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, dilation=1, **cargs):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2D(
            inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias_attr=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias_attr=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Layer):
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, dilation=1, cardinality=1, base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(math.floor(
            outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv1 = nn.Conv2D(inplanes, mid_planes,
                               kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(mid_planes)
        self.conv2 = nn.Conv2D(
            mid_planes, mid_planes, kernel_size=3, stride=stride, padding=dilation,
            bias_attr=False, dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2D(mid_planes)
        self.conv3 = nn.Conv2D(mid_planes, outplanes,
                               kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DlaRoot(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, 1, stride=1, bias_attr=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(paddle.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Layer):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 dilation=1, cardinality=1, base_width=64,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        self.downsample = nn.MaxPool2D(
            stride, stride=stride) if stride > 1 else Identity()
        self.project = Identity()
        cargs = dict(dilation=dilation, cardinality=cardinality,
                     base_width=base_width)
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels,
                              kernel_size=1, stride=1, bias_attr=False),
                    nn.BatchNorm2D(out_channels))
        else:
            cargs.update(dict(root_kernel_size=root_kernel_size,
                              root_residual=root_residual))
            self.tree1 = DlaTree(
                levels - 1, block, in_channels, out_channels, stride, root_dim=0, **cargs)
            self.tree2 = DlaTree(
                levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, **cargs)
        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels,
                                root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Layer):
    def __init__(self, levels, channels, output_stride=32, in_chans=3, cardinality=1,
                 base_width=64, block=DlaBottleneck, residual_root=False,
                 drop_rate=0.0, global_pool='avg', class_dim=1000, with_pool=True):
        super(DLA, self).__init__()
        self.channels = channels
        self.class_dim = class_dim
        self.with_pool = with_pool
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate
        assert output_stride == 32  # FIXME support dilation

        self.base_layer = nn.Sequential(
            nn.Conv2D(in_chans, channels[0], kernel_size=7,
                      stride=1, padding=3, bias_attr=False),
            nn.BatchNorm2D(channels[0]),
            nn.ReLU())
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        cargs = dict(cardinality=cardinality,
                     base_width=base_width, root_residual=residual_root)
        self.level2 = DlaTree(
            levels[2], block, channels[1], channels[2], 2, level_root=False, **cargs)
        self.level3 = DlaTree(
            levels[3], block, channels[2], channels[3], 2, level_root=True, **cargs)
        self.level4 = DlaTree(
            levels[4], block, channels[3], channels[4], 2, level_root=True, **cargs)
        self.level5 = DlaTree(
            levels[5], block, channels[4], channels[5], 2, level_root=True, **cargs)
        self.feature_info = [
            # rare to have a meaningful stride 1 level
            dict(num_chs=channels[0], reduction=1, module='level0'),
            dict(num_chs=channels[1], reduction=2, module='level1'),
            dict(num_chs=channels[2], reduction=4, module='level2'),
            dict(num_chs=channels[3], reduction=8, module='level3'),
            dict(num_chs=channels[4], reduction=16, module='level4'),
            dict(num_chs=channels[5], reduction=32, module='level5'),
        ]

        self.num_features = channels[-1]

        if with_pool:
            self.global_pool = nn.AdaptiveAvgPool2D(1)

        if class_dim > 0:
            self.fc = nn.Conv2D(self.num_features, class_dim, 1)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(mean=0.0, std=math.sqrt(2. / n))
                normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                ones_(m.weight)
                zeros_(m.bias)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1,
                          padding=dilation, bias_attr=False, dilation=dilation),
                nn.BatchNorm2D(planes),
                nn.ReLU()])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward_features(self, x):
        x = self.base_layer(x)
        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.with_pool:
            x = self.global_pool(x)

        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        if self.class_dim > 0:
            x = self.fc(x)
            x = x.flatten(1)

        return x


def dla_34(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 2, 1),
        channels=(16, 32, 64, 128, 256, 512),
        block=DlaBasic,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_34'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_46_c(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 2, 1),
        channels=(16, 32, 64, 64, 128, 256),
        block=DlaBottleneck,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_46_c'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_46x_c(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 2, 1),
        channels=(16, 32, 64, 64, 128, 256),
        block=DlaBottleneck,
        cardinality=32,
        base_width=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_46x_c'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_60x_c(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 3, 1),
        channels=(16, 32, 64, 64, 128, 256),
        block=DlaBottleneck,
        cardinality=32,
        base_width=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_60x_c'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_60(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 3, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_60'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_60x(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 2, 3, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        cardinality=32,
        base_width=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_60x'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_102(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 3, 4, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_102'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_102x(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 3, 4, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        cardinality=32,
        base_width=4,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_102x'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_102x2(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 1, 3, 4, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        cardinality=64,
        base_width=4,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_102x2'])
    if return_transforms:
        return model, transforms
    else:
        return model


def dla_169(pretrained=False, return_transforms=False, **kwargs):
    model = DLA(
        levels=(1, 1, 2, 3, 5, 1),
        channels=(16, 32, 128, 256, 512, 1024),
        block=DlaBottleneck,
        residual_root=True,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls['dla_169'])
    if return_transforms:
        return model, transforms
    else:
        return model
