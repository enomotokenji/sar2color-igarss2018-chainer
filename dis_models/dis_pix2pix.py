import chainer
import chainer.functions as F
import chainer.links as L


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample == 'down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super().__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class ConcatDiscriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 32, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super().__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        #h = F.sum(h, axis=(2, 3))
        return h
