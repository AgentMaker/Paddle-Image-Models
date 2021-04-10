import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ..units import load_model
from .vit import VisionTransformer, Identity, trunc_normal_, zeros_


def get_transforms(resize, crop):
    transforms = T.Compose([
        T.Resize(resize, interpolation='bicubic'),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transforms


transforms_224 = get_transforms(248, 224)
transforms_384 = get_transforms(384, 384)


urls = {
    'deit_ti': r'https://bj.bcebos.com/v1/ai-studio-online/1e91e6ab967b4b0f9940891c6f77f98ca612d5a767b8482498c364c11d65b44b?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_patch16_224.pdparams',
    'deit_s': r'https://bj.bcebos.com/v1/ai-studio-online/56fb3b56543d495aa36cc244e8f25e3e321747cfcedd48c28830ea3a22f4a82a?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_patch16_224.pdparams',
    'deit_b': r'https://bj.bcebos.com/v1/ai-studio-online/38be4cdffc0240c18e9e4905641e9e8171277f42646947e5b3dbcd68c59a6d81?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_224.pdparams',
    'deit_ti_distilled': r'https://bj.bcebos.com/v1/ai-studio-online/dd0ff3e26c1e4fd4b56698a43a62febd35bdc8153563435b898cdd9480cd8720?responseContentDisposition=attachment%3B%20filename%3DDeiT_tiny_distilled_patch16_224.pdparams',
    'deit_s_distilled': r'https://bj.bcebos.com/v1/ai-studio-online/5ab1d5f92e1f44d39db09ab2233143f8fd27788c9b4f46bd9f1d5f2cb760933e?responseContentDisposition=attachment%3B%20filename%3DDeiT_small_distilled_patch16_224.pdparams',
    'deit_b_distilled': r'https://bj.bcebos.com/v1/ai-studio-online/24692c628ab64bfc9bb72fc8a5b3d209080b5ad94227472f98d3bb7cb6732e67?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_224.pdparams',
    'deit_b_384': r'https://bj.bcebos.com/v1/ai-studio-online/de491e7155e94ac2b13b2a97e432155ed6d502e8a0114e4e90ffd6ce9dce63cc?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_patch16_384.pdparams',
    'deit_b_distilled_384': r'https://bj.bcebos.com/v1/ai-studio-online/0a84b9ea45d0412d9bebae9ea3404e679221c3d0c8e542bf9d6a64f810983b25?responseContentDisposition=attachment%3B%20filename%3DDeiT_base_distilled_patch16_384.pdparams'
}


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer=nn.LayerNorm,
                 epsilon=1e-5,
                 class_dim=1000,
                 **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_dim=class_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
            default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        if class_dim > 0:
            self.head_dist = nn.Linear(self.embed_dim, self.class_dim)
            self.head_dist.apply(self._init_weights)

        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)

        if self.class_dim > 0:
            x = self.head(x)
            x_dist = self.head_dist(x_dist)

        return (x + x_dist) / 2


def deit_ti(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_ti'])
    return model, transforms_224


def deit_s(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_s'])
    return model, transforms_224


def deit_b(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_b'])
    return model, transforms_224


def deit_ti_distilled(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_ti_distilled'])
    return model, transforms_224


def deit_s_distilled(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_s_distilled'])
    return model, transforms_224


def deit_b_distilled(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_b_distilled'])
    return model, transforms_224


def deit_b_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_b_384'])
    return model, transforms_384


def deit_b_distilled_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    if pretrained:
        model = load_model(model, urls['deit_b_distilled_384'])
    return model, transforms_384
