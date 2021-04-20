import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ppim.units import load_model
from ppim.models.vit import Mlp, PatchEmbed
from ppim.models.common import add_parameter
from ppim.models.common import DropPath, Identity
from ppim.models.common import trunc_normal_, zeros_, ones_


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
transforms_448 = get_transforms(448, 448)


urls = {
    'cait_xxs_24': r'https://bj.bcebos.com/v1/ai-studio-online/f104732e10e64c48b2848a78b7fa5db45d27a8eed0754c04b367d0708e7242ea?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS24_224.pdparams',
    'cait_xxs_36': r'https://bj.bcebos.com/v1/ai-studio-online/af39ff4c2d6a48faa6dfb901b4fc1de4ae082d767bdc4deb824ae3b600823f1e?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS36_224.pdparams',
    'cait_s_24': r'https://bj.bcebos.com/v1/ai-studio-online/4ecc9cecc89d43cbacf68a0ba14d58a1c9311cc86da3426ab5674fd79827a89a?responseContentDisposition=attachment%3B%20filename%3DCaiT_S24_224.pdparams',
    'cait_xxs_24_384': r'https://bj.bcebos.com/v1/ai-studio-online/0e3615fb421a4301b08fcd675e063a101f4962bad59649f498912123aa0454a4?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS24_384.pdparams',
    'cait_xxs_36_384': r'https://bj.bcebos.com/v1/ai-studio-online/b9f2db8a9c1c43ed971ea4779361c213512ef4c25b664216ab151b6ea60260a7?responseContentDisposition=attachment%3B%20filename%3DCaiT_XXS36_384.pdparams',
    'cait_xs_24_384': r'https://bj.bcebos.com/v1/ai-studio-online/b36139e3caa4427eaaf51aa6de33c8b21f209eef97a44aacb4ec4fe136f93d85?responseContentDisposition=attachment%3B%20filename%3DCaiT_XS24_384.pdparams',
    'cait_s_24_384': r'https://bj.bcebos.com/v1/ai-studio-online/4f57d1db346e435ebb81567399668d6181f054353f6c47e89e9f109b33d724c1?responseContentDisposition=attachment%3B%20filename%3DCaiT_S24_384.pdparams',
    'cait_s_36_384': r'https://bj.bcebos.com/v1/ai-studio-online/445e36df9ec54b23a348bf977b81d92c6f54b31fb28b454d8742e056f99e6417?responseContentDisposition=attachment%3B%20filename%3DCaiT_S36_384.pdparams',
    'cait_m_36_384': r'https://bj.bcebos.com/v1/ai-studio-online/4c73e395068747b9b5c8cdafc3d1b6122a7ed94e6e74481e836eb38c8c46a6eb?responseContentDisposition=attachment%3B%20filename%3DCaiT_M36_384.pdparams',
    'cait_m_48_448': r'https://bj.bcebos.com/v1/ai-studio-online/70515fadc26f48d4b98b33304d8de7c7b955086688324aec8100e5df8a66b15d?responseContentDisposition=attachment%3B%20filename%3DCaiT_M48_448.pdparams'
}


class Class_Attention(nn.Layer):
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(
            (B, 1, self.num_heads, C // self.num_heads)).transpose((0, 2, 1, 3))
        k = self.k(x).reshape((B, N, self.num_heads, C //
                               self.num_heads)).transpose((0, 2, 1, 3))

        q = q * self.scale
        v = self.v(x).reshape((B, N, self.num_heads, C //
                               self.num_heads)).transpose((0, 2, 1, 3))

        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, 1, C))
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls


class LayerScale_Block_CA(nn.Layer):
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6,
                 Attention_block=Class_Attention, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = add_parameter(self, init_values * paddle.ones((dim,)))
        self.gamma_2 = add_parameter(self, init_values * paddle.ones((dim,)))

    def forward(self, x, x_cls):

        u = paddle.concat((x_cls, x), axis=1)

        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))

        x_cls = x_cls + \
            self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class Attention_talking_head(nn.Layer):
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))

        attn = self.proj_l(attn.transpose((0, 2, 3, 1))
                           ).transpose((0, 3, 1, 2))

        attn = nn.functional.softmax(attn, axis=-1)

        attn = self.proj_w(attn.transpose((0, 2, 3, 1))
                           ).transpose((0, 3, 1, 2))
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale_Block(nn.Layer):
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, epsilon=1e-6,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim, epsilon=epsilon)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim, epsilon=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = add_parameter(self, init_values * paddle.ones((dim,)))
        self.gamma_2 = add_parameter(self, init_values * paddle.ones((dim,)))

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class CaiT(nn.Layer):
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, epsilon=1e-6,
                 block_layers=LayerScale_Block, block_layers_token=LayerScale_Block_CA,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU, Attention_block=Attention_talking_head, 
                 Mlp_block=Mlp, init_scale=1e-4, Attention_block_token_only=Class_Attention,
                 Mlp_block_token_only=Mlp, depth_token_only=2, mlp_ratio_clstk=4.0, class_dim=1000):
        super().__init__()

        self.class_dim = class_dim
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = add_parameter(self, paddle.zeros((1, 1, embed_dim)))
        self.pos_embed = add_parameter(
            self, paddle.zeros((1, num_patches, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.LayerList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[
                    i], norm_layer=norm_layer, epsilon=epsilon,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(depth)])

        self.blocks_token_only = nn.LayerList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer, epsilon=epsilon,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale)
            for i in range(depth_token_only)])

        self.norm = norm_layer(embed_dim, epsilon=epsilon)

        # Classifier head
        if class_dim > 0:
            self.head = nn.Linear(embed_dim, class_dim)

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

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
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)

        x = paddle.concat((cls_tokens, x), axis=1)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.class_dim > 0:
            x = self.head(x)

        return x


def cait_xxs_24(pretrained=False, **kwargs):
    model = CaiT(
        img_size=224, embed_dim=192, depth=24,
        num_heads=4, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_xxs_24'])

    return model, transforms_224


def cait_xxs_36(pretrained=False, **kwargs):
    model = CaiT(
        img_size=224, embed_dim=192, depth=36,
        num_heads=4, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_xxs_36'])

    return model, transforms_224


def cait_s_24(pretrained=False, **kwargs):
    model = CaiT(
        img_size=224, embed_dim=384, depth=24,
        num_heads=8, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_s_24'])

    return model, transforms_224


def cait_xxs_24_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=192, depth=24,
        num_heads=4, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_xxs_24_384'])

    return model, transforms_384


def cait_xxs_36_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=192, depth=36,
        num_heads=4, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_xxs_36_384'])

    return model, transforms_384


def cait_xs_24_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=288, depth=24,
        num_heads=6, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_xs_24_384'])

    return model, transforms_384


def cait_s_24_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=384, depth=24,
        num_heads=8, init_scale=1e-5, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_s_24_384'])

    return model, transforms_384


def cait_s_36_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=384, depth=36,
        num_heads=8, init_scale=1e-6, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_s_36_384'])

    return model, transforms_384


def cait_m_36_384(pretrained=False, **kwargs):
    model = CaiT(
        img_size=384, embed_dim=768, depth=36,
        num_heads=16, init_scale=1e-6, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_m_36_384'])

    return model, transforms_384


def cait_m_48_448(pretrained=False, **kwargs):
    model = CaiT(
        img_size=448, embed_dim=768, depth=48,
        num_heads=16, init_scale=1e-6, **kwargs)

    if pretrained:
        model = load_model(model, urls['cait_m_48_448'])

    return model, transforms_448
