import numpy as np

import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ppim.models.common import DropPath, Identity
from ppim.models.common import trunc_normal_, ones_, zeros_
from ppim.models.common import to_2tuple, add_parameter, load_model


def get_transforms(resize, crop):
    transforms = T.Compose(
        [
            T.Resize(resize, interpolation="bicubic"),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms


transforms_224 = get_transforms(248, 224)
transforms_384 = get_transforms(384, 384)
transforms_448 = get_transforms(448, 448)


urls = {
    "lvvit_s": r"https://bj.bcebos.com/v1/ai-studio-online/bf798145d3094d4ab89f99d87a3f99ad576361f3e05e46f4a622de90ef565e9b?responseContentDisposition=attachment%3B%20filename%3Dlvvit_s_224.pdparams",
    "lvvit_m": r"https://bj.bcebos.com/v1/ai-studio-online/c34bcd65d1c94089ab269ffb8927133a7fab39c6a0c44dca8e1c995155cabcd0?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_224.pdparams",
    "lvvit_s_384": r"https://bj.bcebos.com/v1/ai-studio-online/aa4fa51138ea41cb9b413db1308ccc01319f896413764a2d9a3b6e6a23da1ade?responseContentDisposition=attachment%3B%20filename%3Dlvvit_s_384.pdparams",
    "lvvit_m_384": r"https://bj.bcebos.com/v1/ai-studio-online/97d6a53daf55477bbf6e386e00d4763157bcbcea295b402ebb3a26725eaeb772?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_384.pdparams",
    "lvvit_m_448": r"https://bj.bcebos.com/v1/ai-studio-online/b83be46049ac44cfb0821f429e54621020e815f8019944dca81e73a6736b0fdf?responseContentDisposition=attachment%3B%20filename%3Dlvvit_m_448.pdparams",
    "lvvit_l_448": r"https://bj.bcebos.com/v1/ai-studio-online/abd5019da732445eae48ed4eaeff874fc2c00d8d43934ff783d77720b09faef8?responseContentDisposition=attachment%3B%20filename%3Dlvvit_l_448.pdparams",
}


class GroupLinear(nn.Layer):
    """
    Group Linear operator
    """

    def __init__(self, in_planes, out_channels, groups=1, bias=True):
        super(GroupLinear, self).__init__()
        assert in_planes % groups == 0
        assert out_channels % groups == 0
        self.in_dim = in_planes
        self.out_dim = out_channels
        self.groups = groups
        self.group_in_dim = int(self.in_dim / self.groups)
        self.group_out_dim = int(self.out_dim / self.groups)

        self.group_weight = add_parameter(
            self, paddle.zeros((self.groups, self.group_in_dim, self.group_out_dim))
        )

        if bias is True:
            self.group_bias = add_parameter(self, paddle.zeros((self.out_dim,)))
        else:
            self.group_bias = None

    def forward(self, x):
        t, b, d = x.shape
        x = x.reshape((t * b, self.groups, int(d / self.groups)))
        x = x.transpose((1, 0, 2))
        x = paddle.bmm(x, self.group_weight)
        x = x.transpose((1, 0, 2))
        x = x.reshape((t, b, self.out_dim))
        if self.group_bias is not None:
            x = x + self.group_bias

        return x


class Mlp(nn.Layer):
    """
    MLP with support to use group linear operator
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        group=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if group == 1:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc1 = GroupLinear(in_features, hidden_features, group)
            self.fc2 = GroupLinear(hidden_features, out_features, group)
        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    """
    Multi-head self-attention
    with some modification to support different num_heads and head_dim.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            head_dim = dim // num_heads
            self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(
            dim, self.head_dim * self.num_heads * 3, bias_attr=qkv_bias
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape((B, N, 3, self.num_heads, self.head_dim))
            .transpose((2, 0, 3, 1, 4))
        )
        # B,heads,N,C/heads
        q, k, v = qkv[0], qkv[1], qkv[2]

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose((0, 1, 3, 2))

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose((0, 2, 1, 3))
            .reshape((B, N, self.head_dim * self.num_heads))
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    """
    Pre-layernorm transformer block
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.skip_lam = skip_lam

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            group=group,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class MHABlock(nn.Layer):
    """
    Multihead Attention block with residual branch
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.skip_lam = skip_lam
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x * self.skip_lam))) / self.skip_lam
        return x


class FFNBlock(nn.Layer):
    """
    Feed forward network with residual branch
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        group=1,
        skip_lam=1.0,
    ):
        super().__init__()
        self.skip_lam = skip_lam
        self.dim = dim
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            group=group,
        )

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x * self.skip_lam))) / self.skip_lam
        return x


class HybridEmbed(nn.Layer):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768
    ):
        super().__init__()
        assert isinstance(backbone, nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    paddle.zeros((1, in_chans, img_size[0], img_size[1]))
                )[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2D(feature_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.proj(x)
        return x


class PatchEmbedNaive(nn.Layer):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class PatchEmbed4_2(nn.Layer):
    """
    Image to Patch Embedding with 4 layer convolution
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2D(
            in_chans, 64, kernel_size=7, stride=2, padding=3, bias_attr=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            64, 64, kernel_size=3, stride=1, padding=1, bias_attr=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2D(64)
        self.conv3 = nn.Conv2D(
            64, 64, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn3 = nn.BatchNorm2D(64)

        self.proj = nn.Conv2D(
            64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


class PatchEmbed4_2_128(nn.Layer):
    """
    Image to Patch Embedding with 4 layer convolution and 128 filters
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2D(
            in_chans, 128, kernel_size=7, stride=2, padding=3, bias_attr=False
        )  # 112x112
        self.bn1 = nn.BatchNorm2D(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            128, 128, kernel_size=3, stride=1, padding=1, bias_attr=False
        )  # 112x112
        self.bn2 = nn.BatchNorm2D(128)
        self.conv3 = nn.Conv2D(
            128, 128, kernel_size=3, stride=1, padding=1, bias_attr=False
        )
        self.bn3 = nn.BatchNorm2D(128)

        self.proj = nn.Conv2D(
            128, embed_dim, kernel_size=new_patch_size, stride=new_patch_size
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.proj(x)  # [B, C, W, H]

        return x


def get_block(block_type, **kargs):
    if block_type == "mha":
        # multi-head attention block
        return MHABlock(**kargs)
    elif block_type == "ffn":
        # feed forward block
        return FFNBlock(**kargs)
    elif block_type == "tr":
        # transformer block
        return Block(**kargs)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_dpr(drop_path_rate, depth, drop_path_decay="linear"):
    if drop_path_decay == "linear":
        # linear dpr decay
        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, depth)
    elif drop_path_decay == "fix":
        # use fixed dpr
        dpr = [drop_path_rate] * depth
    else:
        # use predefined drop_path_rate list
        assert len(drop_path_rate) == depth
        dpr = drop_path_rate
    return dpr


class LV_ViT(nn.Layer):
    """Vision Transformer with tricks
    Arguements:
        p_emb: different conv based position embedding (default: 4 layer conv)
        skip_lam: residual scalar for skip connection (default: 1.0)
        order: which order of layers will be used (default: None, will override depth if given)
        mix_token: use mix token augmentation for batch of tokens (default: False)
        return_dense: whether to return feature of all tokens with an additional aux_head (default: False)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=3.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_decay="linear",
        hybrid_backbone=None,
        norm_layer=nn.LayerNorm,
        p_emb="4_2",
        head_dim=None,
        skip_lam=1.0,
        order=None,
        mix_token=True,
        return_dense=True,
        class_dim=1000,
    ):
        super().__init__()
        self.class_dim = class_dim
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.output_dim = embed_dim if class_dim == 0 else class_dim

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            if p_emb == "4_2":
                patch_embed_fn = PatchEmbed4_2
            elif p_emb == "4_2_128":
                patch_embed_fn = PatchEmbed4_2_128
            else:
                patch_embed_fn = PatchEmbedNaive

            self.patch_embed = patch_embed_fn(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        num_patches = self.patch_embed.num_patches

        self.cls_token = add_parameter(self, paddle.zeros((1, 1, embed_dim)))
        self.pos_embed = add_parameter(
            self, paddle.zeros((1, num_patches + 1, embed_dim))
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        if order is None:
            dpr = get_dpr(drop_path_rate, depth, drop_path_decay)
            self.blocks = nn.LayerList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        skip_lam=skip_lam,
                    )
                    for i in range(depth)
                ]
            )
        else:
            # use given order to sequentially generate modules
            dpr = get_dpr(drop_path_rate, len(order), drop_path_decay)
            self.blocks = nn.LayerList(
                [
                    get_block(
                        order[i],
                        dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        skip_lam=skip_lam,
                    )
                    for i in range(len(order))
                ]
            )

        self.norm = norm_layer(embed_dim)

        if class_dim > 0:
            self.head = nn.Linear(embed_dim, class_dim)

        self.return_dense = return_dense
        self.mix_token = mix_token

        if (return_dense) and (class_dim > 0):
            self.aux_head = nn.Linear(embed_dim, class_dim)

        if mix_token:
            self.beta = 1.0
            assert return_dense, "always return all features when mixtoken is enabled"

        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, GroupLinear):
            trunc_normal_(m.group_weight)
            if isinstance(m, GroupLinear) and m.group_bias is not None:
                zeros_(m.group_bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_features(self, x):
        # simple forward to obtain feature map (without mixtoken)
        x = self.forward_embeddings(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.forward_tokens(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)

        """
        # Todo...

        # token level mixtoken augmentation
        if self.mix_token and self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2], x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(
                0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        else:
            bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
        """

        bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0

        x = x.flatten(2).transpose((0, 2, 1))
        x = self.forward_tokens(x)

        if self.class_dim > 0:
            x_cls = self.head(x[:, 0])

        if (self.return_dense) and (self.class_dim > 0):
            x_aux = self.aux_head(x[:, 1:])
            return x_cls + 0.5 * x_aux.max(1)[0]

            """
            # Todo...

            if not self.training:
                return x_cls+0.5*x_aux.max(1)[0]

            recover the mixed part
            if self.mix_token and self.training:
                x_aux = x_aux.reshape(
                    x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(
                    0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x
                x_aux = x_aux.reshape(
                    x_aux.shape[0], patch_h*patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
            """
        return x_cls


def lvvit_s(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=224,
        embed_dim=384,
        depth=16,
        num_heads=6,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_s"])
    if return_transforms:
        return model, transforms_224
    else:
        return model


def lvvit_s_384(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=384,
        embed_dim=384,
        depth=16,
        num_heads=6,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_s_384"])
    if return_transforms:
        return model, transforms_384
    else:
        return model


def lvvit_m(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=224,
        embed_dim=512,
        depth=20,
        num_heads=8,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_m"])
    if return_transforms:
        return model, transforms_224
    else:
        return model


def lvvit_m_384(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=384,
        embed_dim=512,
        depth=20,
        num_heads=8,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_m_384"])
    if return_transforms:
        return model, transforms_384
    else:
        return model


def lvvit_m_448(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=448,
        embed_dim=512,
        depth=20,
        num_heads=8,
        p_emb="4_2",
        skip_lam=2.0,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_m_448"])
    if return_transforms:
        return model, transforms_448
    else:
        return model


def lvvit_l_448(pretrained=False, return_transforms=False, **kwargs):
    model = LV_ViT(
        img_size=448,
        embed_dim=768,
        depth=24,
        num_heads=12,
        p_emb="4_2_128",
        skip_lam=3.0,
        order=["tr"] * 24,
        **kwargs,
    )
    if pretrained:
        model = load_model(model, urls["lvvit_l_448"])
    if return_transforms:
        return model, transforms_448
    else:
        return model
