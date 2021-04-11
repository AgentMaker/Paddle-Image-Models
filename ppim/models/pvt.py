import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

import ppim.models.vit as vit

from ppim.units import load_model
from ppim.models.common import trunc_normal_, zeros_, ones_


transforms = T.Compose([
    T.Resize(248, interpolation='bicubic'),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


urls = {
    'pvt_ti': r'https://bj.bcebos.com/v1/ai-studio-online/f833d36454ae4c11be0f5d2eb3041a7e9c2df10b8518434193c0b7c8853dfddf?responseContentDisposition=attachment%3B%20filename%3Dpvt_tiny.pdparams',
    'pvt_s': r'https://bj.bcebos.com/v1/ai-studio-online/608703b1387b44a78d01f09f0c572bd163edecf2354243dda1afeab2b58abb06?responseContentDisposition=attachment%3B%20filename%3Dpvt_small.pdparams',
    'pvt_m': r'https://bj.bcebos.com/v1/ai-studio-online/232d73f40a3b45bb96786a8ae6a58f93967ada580a354266910bb63caa96201b?responseContentDisposition=attachment%3B%20filename%3Dpvt_medium.pdparams',
    'pvt_l': r'https://bj.bcebos.com/v1/ai-studio-online/08b2064702304e13893337d1b1017941ced31fc4f7c644acb4a44a1a81c66e55?responseContentDisposition=attachment%3B%20filename%3Dpvt_large.pdparams'
}


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape((B, N, self.num_heads, C //
                               self.num_heads)).transpose((0, 2, 1, 3))

        if self.sr_ratio > 1:
            x_ = x.transpose((0, 2, 1)).reshape((B, C, H, W))
            x_ = self.sr(x_).reshape((B, C, -1)).transpose((0, 2, 1))
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape((B, -1, 2, self.num_heads, C //
                                      self.num_heads)).transpose((2, 0, 3, 1, 4))
        else:
            kv = self.kv(x).reshape((B, -1, 2, self.num_heads, C //
                                     self.num_heads)).transpose((2, 0, 3, 1, 4))
        k, v = kv[0], kv[1]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(vit.Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6, sr_ratio=1):
        super(Block, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                                    attn_drop, drop_path, act_layer, norm_layer, epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(vit.PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__(
            img_size, patch_size, in_chans, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 epsilon=1e-6, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], class_dim=1000):
        super().__init__()
        self.class_dim = class_dim
        self.depths = depths

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3])

        # pos_embed
        self.pos_embed1 = self.create_parameter(
            shape=(1, self.patch_embed1.num_patches, embed_dims[0]), default_initializer=zeros_)
        self.add_parameter("pos_embed1", self.pos_embed1)
        self.pos_drop1 = nn.Dropout(p=drop_rate)

        self.pos_embed2 = self.create_parameter(
            shape=(1, self.patch_embed2.num_patches, embed_dims[1]), default_initializer=zeros_)
        self.add_parameter("pos_embed2", self.pos_embed2)
        self.pos_drop2 = nn.Dropout(p=drop_rate)

        self.pos_embed3 = self.create_parameter(
            shape=(1, self.patch_embed3.num_patches, embed_dims[2]), default_initializer=zeros_)
        self.add_parameter("pos_embed3", self.pos_embed3)
        self.pos_drop3 = nn.Dropout(p=drop_rate)

        self.pos_embed4 = self.create_parameter(
            shape=(1, self.patch_embed4.num_patches + 1, embed_dims[3]), default_initializer=zeros_)
        self.add_parameter("pos_embed4", self.pos_embed4)
        self.pos_drop4 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = np.linspace(0, drop_path_rate, sum(depths))
        cur = 0
        self.block1 = nn.LayerList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur +
                                                                    i], norm_layer=norm_layer, epsilon=epsilon,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.LayerList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur +
                                                                    i], norm_layer=norm_layer, epsilon=epsilon,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.LayerList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur +
                                                                    i], norm_layer=norm_layer, epsilon=epsilon,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.LayerList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur +
                                                                    i], norm_layer=norm_layer, epsilon=epsilon,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dims[3]), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        # classification head
        if class_dim > 0:
            self.head = nn.Linear(embed_dims[3], class_dim)

        # init weights
        trunc_normal_(self.pos_embed1)
        trunc_normal_(self.pos_embed2)
        trunc_normal_(self.pos_embed3)
        trunc_normal_(self.pos_embed4)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = np.linspace(0, drop_path_rate, sum(self.depths))
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = x.reshape((B, H, W, -1)).transpose((0, 3, 1, 2))

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        x = x + self.pos_embed2
        x = self.pos_drop2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = x.reshape((B, H, W, -1)).transpose((0, 3, 1, 2))

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = x.reshape((B, H, W, -1)).transpose((0, 3, 1, 2))

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for blk in self.block4:
            x = blk(x, H, W)

        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.class_dim > 0:
            x = self.head(x)

        return x


def pvt_ti(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=nn.LayerNorm, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = load_model(model, urls['pvt_ti'])
    return model, transforms


def pvt_s(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = load_model(model, urls['pvt_s'])
    return model, transforms


def pvt_m(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=nn.LayerNorm, depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = load_model(model, urls['pvt_m'])
    return model, transforms


def pvt_l(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=nn.LayerNorm, depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    if pretrained:
        model = load_model(model, urls['pvt_l'])
    return model, transforms
