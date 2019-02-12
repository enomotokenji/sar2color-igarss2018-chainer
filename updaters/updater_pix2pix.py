import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


# Classic Adversarial Loss
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = F.mean(F.softplus(-dis_real))
    L2 = F.mean(F.softplus(dis_fake))
    loss = L1 + L2
    return loss


def loss_dcgan_gen(dis_fake):
    loss = F.mean(F.softplus(-dis_fake))
    return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    return loss


class LossL1:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, t):
        if self.weight == 0:
            return Variable(np.array(0.))

        else:
            return F.mean_absolute_error(x, t) * self.weight


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.models = kwargs.pop('models')
        self.n_dis = kwargs.pop('n_dis')
        self.loss_type = kwargs.pop('loss_type')
        if self.loss_type == 'dcgan':
            self.loss_gen = loss_dcgan_gen
            self.loss_dis = loss_dcgan_dis
        elif self.loss_type == 'hinge':
            self.loss_gen = loss_hinge_gen
            self.loss_dis = loss_hinge_dis
        else:
            raise NotImplementedError
        self.loss_l1 = LossL1(weight=kwargs.pop('weight_l1'))
        super(Updater, self).__init__(*args, **kwargs)

    def get_batch(self, xp):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        paired_sar = []
        paired_rgnir = []
        for j in range(batchsize):
            paired_sar.append(np.asarray(batch[j][0]).astype("f"))
            paired_rgnir.append(np.asarray(batch[j][1]).astype("f"))
        paired_sar = Variable(xp.asarray(paired_sar))
        paired_rgnir = Variable(xp.asarray(paired_rgnir))

        return paired_sar, paired_rgnir

    def update_core(self):
        gen = self.models['gen']
        dis = self.models['dis']
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = gen.xp
        paired_sar, paired_rgnir = self.get_batch(xp)
        for i in range(self.n_dis):
            if i == 0:
                paired_rgnir_fake = gen(paired_sar)
                paired_dis_fake = dis(paired_sar, paired_rgnir_fake)
                loss_gen = self.loss_gen(dis_fake=paired_dis_fake)
                loss_l1 = self.loss_l1(paired_rgnir_fake, paired_rgnir)
                gen.cleargrads()
                loss_gen.backward()
                loss_l1.backward()
                gen_optimizer.update()
                chainer.reporter.report({'loss_gen': loss_gen})
                chainer.reporter.report({'loss_l1': loss_l1})

            paired_sar, paired_rgnir = self.get_batch(xp)
            paired_dis_real = dis(paired_sar, paired_rgnir)
            paired_rgnir_fake = gen(paired_sar)
            paired_dis_fake = dis(paired_sar, paired_rgnir_fake)
            paired_rgnir_fake.unchain_backward()

            loss_dis = self.loss_dis(dis_fake=paired_dis_fake, dis_real=paired_dis_real)
            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()
            chainer.reporter.report({'loss_dis': loss_dis})
