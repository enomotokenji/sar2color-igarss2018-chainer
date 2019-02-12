import math
import chainer
from chainer import functions as F
from chainer import links as L
from chainer.links import Convolution2D

NUMGROUPS = 32


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class Block(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 normalization=None, activation=F.relu, downsample=False):
        super(Block, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.normalization = normalization
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = Convolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = Convolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)
            if self.normalization == 'batchnorm':
                self.b1 = L.BatchNormalization(in_channels)
                self.b2 = L.BatchNormalization(hidden_channels)
            elif self.normalization == 'groupnorm':
                self.b1 = L.GroupNormalization(NUMGROUPS, in_channels)
                self.b2 = L.GroupNormalization(NUMGROUPS, hidden_channels)
            else:
                self.b1 = self.b2 = lambda x: x

    def residual(self, x):
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = Convolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = Convolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
