import numpy as np

import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, KaimingNormal, Constant, Assign


# Common initializations
ones_ = Constant(value=1.0)
zeros_ = Constant(value=0.0)
kaiming_normal_ = KaimingNormal()
trunc_normal_ = TruncatedNormal(std=0.02)


def orthogonal_(tensor, gain=1):
    r"""Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if paddle.fluid.data_feeder.convert_dtype(tensor.dtype) != "float32":
        raise ValueError("Only tensors in float32 dtype are supported")

    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flattened = np.random.randn(rows, cols).astype("float32")

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r, 0)
    ph = paddle.sign(paddle.to_tensor(d))
    q = paddle.to_tensor(q) * ph

    if rows < cols:
        q.t()

    with paddle.no_grad():
        tensor.reshape(q.shape).set_value(q * gain)

    return tensor


# Common Functions
def load_model(model, url):
    path = paddle.utils.download.get_weights_path_from_url(url)
    model.set_state_dict(paddle.load(path))
    return model


def to_2tuple(x):
    return tuple([x] * 2)


def add_parameter(layer, datas, name=None):
    parameter = layer.create_parameter(
        shape=(datas.shape), default_initializer=Assign(datas)
    )
    if name:
        layer.add_parameter(name, parameter)
    return parameter


# Common Layers
def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
