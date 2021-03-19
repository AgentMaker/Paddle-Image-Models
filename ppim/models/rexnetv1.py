import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from math import ceil
from ..units import load_model


transforms = T.Compose([
    T.Resize(256, interpolation='bicubic'),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'rexnet_100': r'https://bj.bcebos.com/v1/ai-studio-online/6c890dd95dfc4e388335adfa298163d3ab413cca558e4abe966d52cb5c3aee31?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.0x.pdparams',
    'rexnet_130': r'https://bj.bcebos.com/v1/ai-studio-online/41a4cc3e6d9545b9b69b4782cafa01147eb7661ec6af4f43841adc734149b3a7?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.3x.pdparams',
    'rexnet_150': r'https://bj.bcebos.com/v1/ai-studio-online/20b131a7cb1840b5aed37c512b2665fb20c72eebe4344da5a3c6f0ab0592a323?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_1.5x.pdparams',
    'rexnet_200': r'https://bj.bcebos.com/v1/ai-studio-online/b4df9f7be43446b0952a25ee6e83f2e443e3b879a00046f6bb33278319cb5fd0?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_2.0x.pdparams',
    'rexnet_300': r'https://bj.bcebos.com/v1/ai-studio-online/9663f0570f0a4e4a8dde0b9799c539f5e22f46917d3d4e5a9d566cd213032d25?responseContentDisposition=attachment%3B%20filename%3Drexnetv1_3.0x.pdparams'
}


def ConvBN(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1, act=None):
    out.append(nn.Conv2D(in_channels, channels, kernel,
                         stride, pad, groups=num_group, bias_attr=False))
    out.append(nn.BatchNorm2D(channels))
    if act == 'swish':
        out.append(Swish())
    elif act == 'relu':
        out.append(nn.ReLU())
    elif act == 'relu6':
        out.append(nn.ReLU6())


class Swish(nn.Layer):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class SE(nn.Layer):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Conv2D(in_channels, channels // se_ratio,
                      kernel_size=1, padding=0),
            nn.BatchNorm2D(channels // se_ratio),
            nn.ReLU(),
            nn.Conv2D(channels // se_ratio, channels,
                      kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Layer):
    def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            ConvBN(out, in_channels=in_channels,
                   channels=dw_channels, act='swish')
        else:
            dw_channels = in_channels

        ConvBN(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
               num_group=dw_channels)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        ConvBN(out, in_channels=dw_channels, channels=channels)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x

        return out


class ReXNetV1(nn.Layer):
    def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0,
                 use_se=True, se_ratio=12, dropout_ratio=0.2, class_dim=1000):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx]
                           for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        ConvBN(features, 3, int(round(stem_channel * width_mult)),
               kernel=3, stride=2, pad=1, act='swish')

        for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(LinearBottleneck(in_channels=in_c,
                                             channels=c,
                                             t=t,
                                             stride=s,
                                             use_se=se, se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        ConvBN(features, c, pen_channels, act='swish')

        features.append(nn.AdaptiveAvgPool2D(1))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2D(pen_channels, class_dim, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x).squeeze()
        return x


def rexnet_100(pretrained=False, **kwargs):
    model = ReXNetV1(width_mult=1.0, **kwargs)
    if pretrained:
        model = load_model(model, urls['rexnet_100'])
    return model, transforms


def rexnet_130(pretrained=False, **kwargs):
    model = ReXNetV1(width_mult=1.3, **kwargs)
    if pretrained:
        model = load_model(model, urls['rexnet_130'])
    return model, transforms


def rexnet_150(pretrained=False, **kwargs):
    model = ReXNetV1(width_mult=1.5, **kwargs)
    if pretrained:
        model = load_model(model, urls['rexnet_150'])
    return model, transforms


def rexnet_200(pretrained=False, **kwargs):
    model = ReXNetV1(width_mult=2.0, **kwargs)
    if pretrained:
        model = load_model(model, urls['rexnet_200'])
    return model, transforms


def rexnet_300(pretrained=False, **kwargs):
    model = ReXNetV1(width_mult=3.0, **kwargs)
    if pretrained:
        model = load_model(model, urls['rexnet_300'])
    return model, transforms
