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
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h


class Generator(chainer.Chain):
    def __init__(self, in_ch, out_ch, normalization=None):
        super().__init__()
        batchnorm = True if normalization else False
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.conv0 = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
            self.cbr1 = CBR(64, 128, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr2 = CBR(128, 256, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr3 = CBR(256, 512, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr4 = CBR(512, 512, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr5 = CBR(512, 512, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr6 = CBR(512, 512, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr7 = CBR(512, 512, bn=batchnorm, sample='down', activation=F.leaky_relu, dropout=False)
            self.cbr8 = CBR(512, 512, bn=batchnorm, sample='up', activation=F.relu, dropout=True)
            self.cbr9 = CBR(1024, 512, bn=batchnorm, sample='up', activation=F.relu, dropout=True)
            self.cbr10 = CBR(1024, 512, bn=batchnorm, sample='up', activation=F.relu, dropout=True)
            self.cbr11 = CBR(1024, 512, bn=batchnorm, sample='up', activation=F.relu, dropout=False)
            self.cbr12 = CBR(1024, 256, bn=batchnorm, sample='up', activation=F.relu, dropout=False)
            self.cbr13 = CBR(512, 128, bn=batchnorm, sample='up', activation=F.relu, dropout=False)
            self.cbr14 = CBR(256, 64, bn=batchnorm, sample='up', activation=F.relu, dropout=False)
            self.conv15 = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x, *args, **kwargs):
        h = x
        h0 = F.leaky_relu(self.conv0(h))
        h1 = self.cbr1(h0)
        h2 = self.cbr2(h1)
        h3 = self.cbr3(h2)
        h4 = self.cbr4(h3)
        h5 = self.cbr5(h4)
        h6 = self.cbr6(h5)
        h = self.cbr7(h6)
        h = F.concat([self.cbr8(h), h6])
        h = F.concat([self.cbr9(h), h5])
        h = F.concat([self.cbr10(h), h4])
        h = F.concat([self.cbr11(h), h3])
        h = F.concat([self.cbr12(h), h2])
        h = F.concat([self.cbr13(h), h1])
        h = F.concat([self.cbr14(h), h0])
        h = self.conv15(h)
        return h
