import chainer
from chainer import functions as F
from chainer import links as L

from gen_models.resblocks import Block, NUMGROUPS


class ResUNetGenerator(chainer.Chain):
    def __init__(self, ch=64, out_ch=3, normalization=None, activation=F.relu, learnable_skip=True, downsample=True):
        super().__init__()
        self.normalization = normalization
        self.act = activation
        self.learnable_skip = learnable_skip
        initializer = chainer.initializers.HeNormal(0.1)
        initializer_skip = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, ch, ksize=3, pad=1, stride=1, initialW=initializer)
            self.block2 = Block(ch, ch * 2, up_or_down='down', normalization=normalization, activation=self.act, downsample=downsample)
            self.block3 = Block(ch * 2, ch * 4, up_or_down='down', normalization=normalization, activation=self.act, downsample=downsample)
            self.block4 = Block(ch * 4, ch * 8, up_or_down='down', normalization=normalization, activation=self.act, downsample=downsample)
            self.block5 = Block(ch * 8, ch * 16, up_or_down='down', normalization=normalization, activation=self.act, downsample=downsample)
            self.block6 = Block(ch * 16, ch * 8, up_or_down='up', normalization=normalization, activation=self.act)
            self.block7 = Block(ch * 8, ch * 4, up_or_down='up', normalization=normalization, activation=self.act)
            self.block8 = Block(ch * 4, ch * 2, up_or_down='up', normalization=normalization, activation=self.act)
            self.block9 = Block(ch * 2, ch, up_or_down='up', normalization=normalization, activation=self.act)
            self.conv10 = L.Convolution2D(ch, out_ch, ksize=3, pad=1, stride=1, initialW=initializer)

            if self.learnable_skip:
                self.c6_skip = L.Convolution2D(ch * 8, ch * 8, ksize=1, pad=0, initialW=initializer_skip)
                self.c7_skip = L.Convolution2D(ch * 4, ch * 4, ksize=1, pad=0, initialW=initializer_skip)
                self.c8_skip = L.Convolution2D(ch * 2, ch * 2, ksize=1, pad=0, initialW=initializer_skip)
                self.c9_skip = L.Convolution2D(ch, ch, ksize=1, pad=0, initialW=initializer_skip)
            else:
                self.c6_skip = self.c7_skip = self.c8_skip = self.c9 = lambda x: x

            if self.normalization == 'batchnorm':
                self.b10 = L.BatchNormalization(ch)
            elif self.normalization == 'groupnorm':
                self.b10 = L.GroupNormalization(NUMGROUPS, ch)
            elif self.normalization is None:
                self.b10 = lambda x: x
            else:
                raise NotImplementedError

    def __call__(self, x):
        h = x
        h1 = self.conv1(h)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h = self.block5(h4)

        h = self.block6(h) + self.c6_skip(h4)
        h = self.block7(h) + self.c7_skip(h3)
        h = self.block8(h) + self.c8_skip(h2)
        h = self.block9(h) + self.c9_skip(h1)
        h = self.b10(h)
        h = self.act(h)
        h = F.tanh(self.conv10(h))
        return h
