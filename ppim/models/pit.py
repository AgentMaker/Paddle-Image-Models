import math
import paddle

from paddle import nn
from paddle.nn.initializer import Assign

from ..units import load_model
from .vit import trunc_normal_, zeros_, ones_, Block


transforms = T.Compose([
    T.Resize(248, interpolation='bicubic'),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'pit_ti': 'https://bj.bcebos.com/v1/ai-studio-online/1e91e6ab967b4b0f9940891c6f77f98ca612d5a767b8482498c364c11d65b44b?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_patch16_224.pdparams',
    'pit_xs': 'https://bj.bcebos.com/v1/ai-studio-online/56fb3b56543d495aa36cc244e8f25e3e321747cfcedd48c28830ea3a22f4a82a?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_patch16_224.pdparams',
    'pit_s': 'https://bj.bcebos.com/v1/ai-studio-online/38be4cdffc0240c18e9e4905641e9e8171277f42646947e5b3dbcd68c59a6d81?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_224.pdparams',
    'pit_b': 'https://bj.bcebos.com/v1/ai-studio-online/de491e7155e94ac2b13b2a97e432155ed6d502e8a0114e4e90ffd6ce9dce63cc?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_384.pdparams',
    'pit_ti_distilled': 'https://bj.bcebos.com/v1/ai-studio-online/dd0ff3e26c1e4fd4b56698a43a62febd35bdc8153563435b898cdd9480cd8720?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_distilled_patch16_224.pdparams',
    'pit_xs_distilled': 'https://bj.bcebos.com/v1/ai-studio-online/5ab1d5f92e1f44d39db09ab2233143f8fd27788c9b4f46bd9f1d5f2cb760933e?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_distilled_patch16_224.pdparams',
    'pit_s_distilled': 'https://bj.bcebos.com/v1/ai-studio-online/24692c628ab64bfc9bb72fc8a5b3d209080b5ad94227472f98d3bb7cb6732e67?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_224.pdparams',
    'pit_b_distilled': 'https://bj.bcebos.com/v1/ai-studio-online/0a84b9ea45d0412d9bebae9ea3404e679221c3d0c8e542bf9d6a64f810983b25?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_384.pdparams'
}


class Transformer(nn.Layer):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.LayerList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer='nn.LayerNorm',
                epsilon=1e-6,
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        n, c, h, w = x.shape
        # x = rearrange(x, 'b c h w -> b (h w) c')
        x = x.transpose((0, 2, 3, 1))
        x = paddle.flatten(x, start_axis=1, stop_axis=2)

        token_length = cls_tokens.shape[1]
        x = paddle.concat((cls_tokens, x), axis=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = x.transpose((0, 2, 1))
        x = x.reshape((n, c, h, w))
        return x, cls_tokens


class conv_head_pooling(nn.Layer):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2D(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Layer):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias_attr=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Layer):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, in_chans=3, attn_drop_rate=.0, drop_rate=.0,
                 drop_path_rate=.0, class_dim=1000):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)

        self.base_dims = base_dims
        self.heads = heads
        self.class_dim = class_dim

        self.patch_size = patch_size

        self.pos_embed = self.create_parameter(
            shape=(1, base_dims[0] * heads[0], width, width),
            default_initializer=Assign(
                paddle.randn((1, base_dims[0] * heads[0], width, width))
            ))
        self.add_parameter("pos_embed", self.pos_embed)

        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.cls_token = self.create_parameter(
            shape=(1, 1, base_dims[0] * heads[0]),
            default_initializer=Assign(
                paddle.randn((1, 1, base_dims[0] * heads[0]))
            ))
        self.add_parameter("cls_token", self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.LayerList([])
        self.pools = nn.LayerList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio, drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
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
        x, cls_tokens = self.transformers[len(
            self.transformers)-1](x, cls_tokens)

        cls_tokens = self.norm(cls_tokens)

        return cls_tokens

    def forward(self, x):
        cls_token = self.forward_features(x)

        if self.class_dim > 0:
            cls_token = self.head(cls_token[:, 0])

        return cls_token


class DistilledPoolingTransformer(PoolingTransformer):
    def __init__(self, *args, **kwargs):
        super(DistilledPoolingTransformer, self).__init__(*args, **kwargs)
        self.cls_token = self.create_parameter(
            shape=(1, 2, self.base_dims[0] * self.heads[0]),
            default_initializer=Assign(
                paddle.randn((1, 2, self.base_dims[0] * self.heads[0]))
            ))
        self.add_parameter("cls_token", self.cls_token)

        if self.class_dim > 0:
            self.head_dist = nn.Linear(self.base_dims[-1] * self.heads[-1],
                                       self.class_dim)

        trunc_normal_(self.cls_token)
        self.head_dist.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.forward_features(x)
        x_cls, x_dist = cls_token[:, 0], cls_token[:, 1]

        if self.class_dim > 0:
            x_cls = self.head(x_cls)
            x_dist = self.head_dist(x_dist)

        return (x_cls + x_dist) / 2


def pit_b(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_b'])
    return model


def pit_s(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_s'])
    return model


def pit_xs(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_xs'])
    return model


def pit_ti(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_ti'])
    return model


def pit_b_distilled(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_b_distilled'])
    return model


def pit_s_distilled(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_s_distilled'])
    return model


def pit_xs_distilled(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_xs_distilled'])
    return model


def pit_ti_distilled(pretrained, **kwargs):
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
        model = load_model(model, urls['pit_ti_distilled'])
    return model
