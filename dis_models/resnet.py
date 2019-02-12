import chainer
from chainer import functions as F
from chainer.links import Linear
from dis_models.resblocks import Block, OptimizedBlock


class ConcatResNetDiscriminator(chainer.Chain):
    def __init__(self, x_channels=1, y_channels=3, ch=64, activation=F.relu):
        super().__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block0_0 = OptimizedBlock(x_channels, ch // 2)
            self.block0_1 = OptimizedBlock(y_channels, ch // 2)
            self.block1 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block2 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block3 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block4 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block5 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l6 = Linear(ch * 16, 1, initialW=initializer)

    def __call__(self, x_0, x_1):
        h = F.concat([self.block0_0(x_0), self.block0_1(x_1)])
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l6(h)
        return output
