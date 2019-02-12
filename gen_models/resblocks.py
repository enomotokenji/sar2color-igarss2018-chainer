import math
import chainer
import chainer.links as L
from chainer import functions as F

NUMGROUPS = 32


def _downsample(x):
    return F.average_pooling_2d(x, 2)


def downsample_conv(x, conv):
    return conv(_downsample(x))


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def upsample_conv(x, conv):
    return conv(_upsample(x))


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 normalization=None, activation=F.relu, up_or_down=None, downsample=True, dropout=False):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.normalization = normalization
        self.activation = activation
        self.learnable_sc = in_channels != out_channels or (up_or_down == 'up' or 'down')
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.up_or_down = up_or_down
        if self.up_or_down == 'up':
            ksize, pad, stride = 3, 1, 1
            self._up_or_down_sample_conv = upsample_conv
        elif self.up_or_down == 'down':
            if downsample:
                ksize, pad, stride = 3, 1, 1
                self._up_or_down_sample_conv = downsample_conv
            else:
                ksize, pad, stride = 2, 0, 2
                self._up_or_down_sample_conv = lambda x, c: c(x)
        elif self.up_or_down is None:
            ksize, pad, stride = 3, 1, 1
            self._up_or_down_sample_conv = lambda x, c: c(x)
        else:
            raise NotImplementedError
        self.dropout = F.dropout if dropout else lambda x: x

        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, stride=stride, initialW=initializer)
            self.c2 = L.Convolution2D(hidden_channels, out_channels, ksize=3, pad=1, initialW=initializer)
            if self.normalization == 'batchnorm':
                self.b1 = L.BatchNormalization(in_channels)
                self.b2 = L.BatchNormalization(hidden_channels)
            elif self.normalization == 'groupnorm':
                self.b1 = L.GroupNormalization(NUMGROUPS, in_channels)
                self.b2 = L.GroupNormalization(NUMGROUPS, hidden_channels)
            else:
                self.b1 = self.b2 = lambda x: x
            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad, stride=stride, initialW=initializer_sc)

    def residual(self, x, **kwargs):
        h = x
        h = self.b1(h, **kwargs)
        h = self.activation(h)
        h = self._up_or_down_sample_conv(h, self.c1)
        h = self.b2(h, **kwargs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self._up_or_down_sample_conv(x, self.c_sc)
            return x
        else:
            return x

    def __call__(self, x, **kwargs):
        return self.residual(x, **kwargs) + self.shortcut(x)
