import paddle
import paddle.nn as nn
import paddle.vision.transforms as T

from ppim.models.common import load_model


transforms = T.Compose(
    [
        T.Resize(256, interpolation="bilinear"),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


urls = {
    "mixer_b": r"https://bj.bcebos.com/v1/ai-studio-online/8fcd0b6ba98042d68763bbcbfe96375cbfd97ffed8334ac09787ef73ecf9989f?responseContentDisposition=attachment%3B%20filename%3Dimagenet1k_Mixer-B_16.pdparams",
    "mixer_l": r"https://bj.bcebos.com/v1/ai-studio-online/ca74ababd4834e34b089c1485989738de4fdf6a97be645ed81b6e39449c5815c?responseContentDisposition=attachment%3B%20filename%3Dimagenet1k_Mixer-L_16.pdparams",
}


class MlpBlock(nn.Layer):
    def __init__(self, features_dim, mlp_dim):
        super().__init__()
        self.fc_0 = nn.Linear(features_dim, mlp_dim)
        self.fc_1 = nn.Linear(mlp_dim, features_dim)

    def forward(self, x):
        y = self.fc_0(x)
        y = nn.functional.gelu(y)
        y = self.fc_1(y)
        return y


class MixerBlock(nn.Layer):
    def __init__(
        self,
        token_dim,
        channels_dim,
        tokens_mlp_dim,
        channels_mlp_dim,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm_0 = norm_layer(channels_dim, epsilon=epsilon)
        self.token_mixing = MlpBlock(token_dim, tokens_mlp_dim)
        self.norm_1 = norm_layer(channels_dim, epsilon=epsilon)
        self.channel_mixing = MlpBlock(channels_dim, channels_mlp_dim)

    def forward(self, x):
        y = self.norm_0(x)
        y = y.transpose((0, 2, 1))
        y = self.token_mixing(y)
        y = y.transpose((0, 2, 1))
        x = x + y
        y = self.norm_1(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MlpMixer(nn.Layer):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        num_blocks=12,
        hidden_dim=768,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
        class_dim=1000,
    ):
        super().__init__()
        self.class_dim = class_dim

        self.stem = nn.Conv2D(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        blocks = [
            MixerBlock(
                (img_size[0] // patch_size[0]) ** 2,
                hidden_dim,
                tokens_mlp_dim,
                channels_mlp_dim,
                norm_layer,
                epsilon,
            )
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.pre_head_layer_norm = norm_layer(hidden_dim, epsilon=epsilon)

        if class_dim > 0:
            self.head = nn.Linear(hidden_dim, class_dim)

    def forward(self, inputs):
        x = self.stem(inputs)

        x = x.transpose((0, 2, 3, 1))
        x = x.flatten(1, 2)

        x = self.blocks(x)
        x = self.pre_head_layer_norm(x)

        if self.class_dim > 0:
            x = x.mean(axis=1)
            x = self.head(x)

        return x


def mixer_b(pretrained=False, return_transforms=False, **kwargs):
    model = MlpMixer(
        hidden_dim=768,
        num_blocks=12,
        tokens_mlp_dim=384,
        channels_mlp_dim=3072,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["mixer_b"])
    if return_transforms:
        return model, transforms
    else:
        return model


def mixer_l(pretrained=False, return_transforms=False, **kwargs):
    model = MlpMixer(
        hidden_dim=1024,
        num_blocks=24,
        tokens_mlp_dim=512,
        channels_mlp_dim=4096,
        **kwargs
    )
    if pretrained:
        model = load_model(model, urls["mixer_l"])
    if return_transforms:
        return model, transforms
    else:
        return model
