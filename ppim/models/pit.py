import math

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ppim.models.vit import Block

from ppim.models.common import add_parameter, load_model
from ppim.models.common import trunc_normal_, zeros_, ones_


transforms = T.Compose(
    [
        T.Resize(248, interpolation="bicubic"),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


urls = {
    "pit_ti": r"https://bj.bcebos.com/v1/ai-studio-online/3d0fe9a33bb74abaa0648f6200b37e5b49ca9a4f15a04afbab7a885da64dfa62?responseContentDisposition=attachment%3B%20filename%3Dpit_ti.pdparams",
    "pit_xs": r"https://bj.bcebos.com/v1/ai-studio-online/4bee539cc81a477a8bae4795f91d583c810ea4832e6d4ed983b37883669e6a6d?responseContentDisposition=attachment%3B%20filename%3Dpit_xs.pdparams",
    "pit_s": r"https://bj.bcebos.com/v1/ai-studio-online/232c216331d04fb58f77839673b34652ea229a9ab84044a493e08cd802ab4fe3?responseContentDisposition=attachment%3B%20filename%3Dpit_s.pdparams",
    "pit_b": r"https://bj.bcebos.com/v1/ai-studio-online/26f33b44d9424626b74eb7cfad2041582afabdebd6474afa976cc0a55c226791?responseContentDisposition=attachment%3B%20filename%3Dpit_b.pdparams",
    "pit_ti_distilled": r"https://bj.bcebos.com/v1/ai-studio-online/9707c73717274b5e880e8401b85dcf9ad12b0d7e47944af68b3d6a2236b70567?responseContentDisposition=attachment%3B%20filename%3Dpit_ti_distill.pdparams",
    "pit_xs_distilled": r"https://bj.bcebos.com/v1/ai-studio-online/61aa3339366d4315854bf67a8df1cea20f4a2402b2d94d7688d995423a197df1?responseContentDisposition=attachment%3B%20filename%3Dpit_xs_distill.pdparams",
    "pit_s_distilled": r"https://bj.bcebos.com/v1/ai-studio-online/65acbfa1d6a94c689225fe95c6ec48567f5c05ee051243d6abe3bbcbd6119f5d?responseContentDisposition=attachment%3B%20filename%3Dpit_s_distill.pdparams",
    "pit_b_distilled": r"https://bj.bcebos.com/v1/ai-studio-online/2d6631b21542486b8333440c612847f35a7782d2890f4514ad8007c34ae77e66?responseContentDisposition=attachment%3B%20filename%3Dpit_b_distill.pdparams",
}


class Transformer(nn.Layer):
    def __init__(
        self,
        base_dim,
        depth,
        heads,
        mlp_ratio,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_prob=None,
    ):
        super(Transformer, self).__init__()
        self.layers = nn.LayerList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_prob[i],
                    norm_layer=nn.LayerNorm,
                    epsilon=1e-6,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, cls_tokens):
        n, c, h, w = x.shape
        x = x.transpose((0, 2, 3, 1))
        x = paddle.flatten(x, start_axis=1, stop_axis=2)

        token_length = cls_tokens.shape[1]
        x = paddle.concat((cls_tokens, x), axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = x.transpose((0, 2, 1))
        x = x.reshape((n, c, h, w))
        return x, cls_tokens


class conv_head_pooling(nn.Layer):
    def __init__(self, in_feature, out_feature, stride, padding_mode="zeros"):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2D(
            in_feature,
            out_feature,
            kernel_size=stride + 1,
            padding=stride // 2,
            stride=stride,
            padding_mode=padding_mode,
            groups=in_feature,
        )
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Layer):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
            bias_attr=True,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        stride,
        base_dims,
        depth,
        heads,
        mlp_ratio,
        in_chans=3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        class_dim=1000,
    ):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor((image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.class_dim = class_dim

        self.patch_size = patch_size

        self.pos_embed = add_parameter(
            self, paddle.randn((1, base_dims[0] * heads[0], width, width))
        )

        self.patch_embed = conv_embedding(
            in_chans, base_dims[0] * heads[0], patch_size, stride, padding
        )

        self.cls_token = add_parameter(
            self, paddle.randn((1, 1, base_dims[0] * heads[0]))
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.LayerList([])
        self.pools = nn.LayerList([])

        for stage in range(len(depth)):
            drop_path_prob = [
                drop_path_rate * i / total_block
                for i in range(block_idx, block_idx + depth[stage])
            ]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(
                    base_dims[stage],
                    depth[stage],
                    heads[stage],
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_prob,
                )
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(
                        base_dims[stage] * heads[stage],
                        base_dims[stage + 1] * heads[stage + 1],
                        stride=2,
                    )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], epsilon=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if class_dim > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], class_dim)

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)

        pos_embed = self.pos_embed
        x = self.pos_drop(x + pos_embed)
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))

        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[len(self.transformers) - 1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)[:, 0]

        if self.class_dim > 0:
            cls_token = self.head(cls_token)

        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super(DistilledPoolingTransformer, self).__init__(*args, **kwargs)
        self.cls_token = add_parameter(
            self, paddle.randn((1, 2, self.base_dims[0] * self.heads[0]))
        )

        if self.class_dim > 0:
            self.head_dist = nn.Linear(
                self.base_dims[-1] * self.heads[-1], self.class_dim
            )

        trunc_normal_(self.cls_token)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls, x_dist = cls_token[:, 0], cls_token[:, 1]

        if self.class_dim > 0:
            x_cls = self.head(x_cls)
            x_dist = self.head_dist(x_dist)

        return (x_cls + x_dist) / 2


def pit_b(pretrained=False, return_transforms=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_b"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_s(pretrained=False, return_transforms=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_s"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_xs(pretrained=False, return_transforms=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_xs"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_ti(pretrained=False, return_transforms=False, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_ti"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_b_distilled(pretrained=False, return_transforms=False, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_b_distilled"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_s_distilled(pretrained=False, return_transforms=False, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_s_distilled"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_xs_distilled(pretrained=False, return_transforms=False, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_xs_distilled"])
    if return_transforms:
        return model, transforms
    else:
        return model


def pit_ti_distilled(pretrained=False, return_transforms=False, **kwargs):
    model = DistilledPoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[32, 32, 32],
        depth=[2, 6, 4],
        heads=[2, 4, 8],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["pit_ti_distilled"])
    if return_transforms:
        return model, transforms
    else:
        return model
