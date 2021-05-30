import itertools
import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ppim.models.common import zeros_
from ppim.models.common import load_model


transforms = T.Compose([
    T.Resize(248, interpolation='bicubic'),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'levit_128s': r'https://bj.bcebos.com/v1/ai-studio-online/3d875a55328c487d833363574b3e6b3be41be4b44a8542dc972f089a6f9e7cc5?responseContentDisposition=attachment%3B%20filename%3DLeViT-128S.pdparams',
    'levit_128': r'https://bj.bcebos.com/v1/ai-studio-online/8cf41e96d369411dbc526177c040d68b041c2fd282674069b89a51b16431e3db?responseContentDisposition=attachment%3B%20filename%3DLeViT-128.pdparams',
    'levit_192': r'https://bj.bcebos.com/v1/ai-studio-online/26275887b63b4fe9bea2d5155229e49ba5ee6e49f3294e1c8c88eec08f19cd09?responseContentDisposition=attachment%3B%20filename%3DLeViT-192.pdparams',
    'levit_256': r'https://bj.bcebos.com/v1/ai-studio-online/9c869ecde73147b39726ba6a154e91ef326de0a04b2d4ad5809a7a50db7a6ea0?responseContentDisposition=attachment%3B%20filename%3DLeViT-256.pdparams',
    'levit_384': r'https://bj.bcebos.com/v1/ai-studio-online/1e39ab61d8a5408aa08cdb30a9de52f9e17b839bd3054be7b8b2c6de1e19742c?responseContentDisposition=attachment%3B%20filename%3DLeViT-384.pdparams',
}


class Conv2D_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_sublayer(
            'c', nn.Conv2D(
                a, b, ks, stride, pad, dilation, groups, bias_attr=False
            )
        )

        bn = nn.BatchNorm2D(b)

        Constant(bn_weight_init)(bn.weight)
        zeros_(bn.bias)

        self.add_sublayer('bn', bn)


class Linear_BN(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, resolution=-100000):
        super().__init__()
        self.add_sublayer('c', nn.Linear(a, b, bias_attr=False))

        bn = nn.BatchNorm1D(b)

        Constant(bn_weight_init)(bn.weight)
        zeros_(bn.bias)

        self.add_sublayer('bn', bn)

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        x = bn(x.flatten(0, 1)).reshape(x.shape)
        return x


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias_attr=True, std=0.02):
        super().__init__()
        self.add_sublayer('bn', nn.BatchNorm1D(a))

        l = nn.Linear(a, b, bias_attr=bias_attr)

        TruncatedNormal(std=std)(l.weight)

        if bias_attr:
            zeros_(l.bias)

        self.add_sublayer('l', l)


class Residual(nn.Layer):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * paddle.to_tensor((np.random.rand(x.shape[0], 1, 1) > self.drop) / (1 - self.drop))
        else:
            return x + self.m(x)


def b16(n, activation=nn.Hardswish, resolution=224):
    return nn.Sequential(
        Conv2D_BN(3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2D_BN(n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2D_BN(n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2D_BN(n // 2, n, 3, 2, 1, resolution=resolution // 8)
    )


class Attention(nn.Layer):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 activation=nn.Hardswish,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.qkv = Linear_BN(dim, h, resolution=resolution)
        self.proj = nn.Sequential(
            activation(),
            Linear_BN(
                self.dh, dim, bn_weight_init=0, resolution=resolution
            )
        )

        points = list(itertools.product(range(resolution), range(resolution)))
        self.N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = self.create_parameter(
            (num_heads, len(attention_offsets)),
            default_initializer=zeros_
        )
        self.register_buffer(
            'attention_bias_idxs',
            paddle.to_tensor(idxs, dtype='int64')
        )

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.reshape(
            (B, N, self.num_heads, -1)
        ).split(
            [self.key_dim, self.key_dim, self.d], axis=3
        )

        q = q.transpose((0, 2, 1, 3))
        k = k.transpose((0, 2, 1, 3))
        v = v.transpose((0, 2, 1, 3))

        attn = (
            (q @ k.transpose((0, 1, 3, 2))) * self.scale +
            paddle.index_select(
                self.attention_biases, self.attention_bias_idxs, axis=1
            ).reshape(
                (self.num_heads, self.N, self.N)
            )
        )
        attn = nn.functional.softmax(attn, axis=-1)
        x = (attn @ v).transpose((0, 2, 1, 3)).reshape((B, N, self.dh))
        x = self.proj(x)
        return x


class Subsample(nn.Layer):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(
            (B, self.resolution, self.resolution, C)
        )[:, ::self.stride, ::self.stride].reshape((B, -1, C))
        return x


class AttentionSubsample(nn.Layer):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8,
                 attn_ratio=2,
                 activation=nn.Hardswish,
                 stride=2,
                 resolution=14, resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        h = self.dh + nh_kd

        self.kv = Linear_BN(in_dim, h, resolution=resolution)
        self.q = nn.Sequential(
            Subsample(stride, resolution),
            Linear_BN(in_dim, nh_kd, resolution=resolution_))
        self.proj = nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim, resolution=resolution_))

        self.stride = stride
        self.resolution = resolution

        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(itertools.product(
            range(resolution_), range(resolution_)))
        self.N = len(points)
        self.N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                    abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = self.create_parameter(
            (num_heads, len(attention_offsets)),
            default_initializer=zeros_
        )
        self.register_buffer(
            'attention_bias_idxs',
            paddle.to_tensor(idxs, dtype='int64')
        )

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).reshape((B, N, self.num_heads, -
                                   1)).split([self.key_dim, self.d], axis=3)
        k = k.transpose((0, 2, 1, 3))  # BHNC
        v = v.transpose((0, 2, 1, 3))  # BHNC
        q = self.q(x).reshape((B, self.resolution_2, self.num_heads,
                               self.key_dim)).transpose((0, 2, 1, 3))

        attn = (
            (q @ k.transpose((0, 1, 3, 2))) * self.scale +
            paddle.index_select(
                self.attention_biases,
                self.attention_bias_idxs,
                axis=1
            ).reshape((self.num_heads, self.N_, self.N))
        )

        attn = nn.functional.softmax(attn, axis=-1)

        x = (attn @ v).transpose((0, 2, 1, 3)).reshape((B, -1, self.dh))
        x = self.proj(x)
        return x


class LeViT(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, embed_dim=[128, 256, 384], key_dim=[16, 16, 16],
                 depth=[2, 3, 4], num_heads=[4, 6, 8], attn_ratio=[2, 2, 2], mlp_ratio=[2, 2, 2],
                 down_ops=[['Subsample', 16, 128 // 16, 4, 2, 2],
                           ['Subsample', 16, 256 // 16, 4, 2, 2]],
                 attention_activation=nn.Hardswish, mlp_activation=nn.Hardswish,
                 hybrid_backbone=b16(128, activation=nn.Hardswish),
                 distillation=True, drop_path=0, class_dim=1000):
        super().__init__()
        self.class_dim = class_dim
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation
        self.patch_embed = hybrid_backbone
        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size

        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio, mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(Attention(
                        ed, kd, nh,
                        attn_ratio=ar,
                        activation=attention_activation,
                        resolution=resolution,
                    ), drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(nn.Sequential(
                            Linear_BN(ed, h, resolution=resolution),
                            mlp_activation(),
                            Linear_BN(h, ed, bn_weight_init=0,
                                      resolution=resolution),
                        ), drop_path))

            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2], key_dim=do[1], num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(nn.Sequential(
                            Linear_BN(embed_dim[i + 1], h,
                                      resolution=resolution),
                            mlp_activation(),
                            Linear_BN(
                                h, embed_dim[i + 1], bn_weight_init=0, resolution=resolution),
                        ), drop_path))
        self.blocks = nn.Sequential(*self.blocks)

        # Classifier head
        if class_dim > 0:
            self.head = BN_Linear(embed_dim[-1], class_dim)
            if distillation:
                self.head_dist = BN_Linear(embed_dim[-1], class_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose((0, 2, 1))
        x = self.blocks(x)

        if self.class_dim > 0:
            x = x.mean(1)
            if self.distillation:
                x = self.head(x), self.head_dist(x)
                x = (x[0] + x[1]) / 2
            else:
                x = self.head(x)

        return x


def levit_128s(pretrained=False, return_transforms=False, **kwarg):
    model = LeViT(
        embed_dim=[128, 256, 384],
        num_heads=[4, 6, 8],
        key_dim=[16, 16, 16],
        depth=[2, 3, 4],
        down_ops=[
            ['Subsample', 16, 128 // 16, 4, 2, 2],
            ['Subsample', 16, 256 // 16, 4, 2, 2],
        ],
        hybrid_backbone=b16(128),
        **kwarg
    )
    if pretrained:
        model = load_model(model, urls['levit_128s'])
    if return_transforms:
        return model, transforms
    else:
        return model


def levit_128(pretrained=False, return_transforms=False, **kwarg):
    model = LeViT(
        embed_dim=[128, 256, 384],
        num_heads=[4, 8, 12],
        key_dim=[16, 16, 16],
        depth=[4, 4, 4],
        down_ops=[
            ['Subsample', 16, 128 // 16, 4, 2, 2],
            ['Subsample', 16, 256 // 16, 4, 2, 2],
        ],
        hybrid_backbone=b16(128),
        **kwarg
    )
    if pretrained:
        model = load_model(model, urls['levit_128'])
    if return_transforms:
        return model, transforms
    else:
        return model


def levit_192(pretrained=False, return_transforms=False, **kwarg):
    model = LeViT(
        embed_dim=[192, 288, 384],
        num_heads=[3, 5, 6],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        down_ops=[
            ['Subsample', 32, 192 // 32, 4, 2, 2],
            ['Subsample', 32, 288 // 32, 4, 2, 2],
        ],
        hybrid_backbone=b16(192),
        **kwarg
    )
    if pretrained:
        model = load_model(model, urls['levit_192'])
    if return_transforms:
        return model, transforms
    else:
        return model


def levit_256(pretrained=False, return_transforms=False, **kwarg):
    model = LeViT(
        embed_dim=[256, 384, 512],
        num_heads=[4, 6, 8],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        down_ops=[
            ['Subsample', 32, 256 // 32, 4, 2, 2],
            ['Subsample', 32, 384 // 32, 4, 2, 2],
        ],
        hybrid_backbone=b16(256)
    )
    if pretrained:
        model = load_model(model, urls['levit_256'])
    if return_transforms:
        return model, transforms
    else:
        return model


def levit_384(pretrained=False, return_transforms=False, **kwarg):
    model = LeViT(
        embed_dim=[384, 512, 768],
        num_heads=[6, 9, 12],
        key_dim=[32, 32, 32],
        depth=[4, 4, 4],
        down_ops=[
            ['Subsample', 32, 384 // 32, 4, 2, 2],
            ['Subsample', 32, 512 // 32, 4, 2, 2],
        ],
        hybrid_backbone=b16(384)
    )
    if pretrained:
        model = load_model(model, urls['levit_384'])
    if return_transforms:
        return model, transforms
    else:
        return model
