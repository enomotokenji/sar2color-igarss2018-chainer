import cv2
import random

import chainer

from datasets.utils import read_imlist, train_test_split_imlist


def paired_sar_rgnir(*args, **kwargs):
    paired_sar = read_imlist(kwargs.pop('dir_paired_sar'), kwargs.pop('imlist_paired_sar'))
    paired_rgnir = read_imlist(kwargs.pop('dir_paired_rgnir'), kwargs.pop('imlist_paired_rgnir'))
    if not len(paired_sar) == len(paired_rgnir):
        raise Exception('The length of paired rgnir list and that of paired sar list must be same.')

    train_paired_sar, test_paired_sar, train_paired_rgnir, test_paired_rgnir = train_test_split_imlist(paired_sar, paired_rgnir, kwargs.pop('rate_train'))

    train = PairedSARRGNIR(train_paired_rgnir, train_paired_sar, *args, **kwargs)
    kwargs.update({'augmentation': False, 'size': 256})
    test = PairedSARRGNIR(test_paired_rgnir, test_paired_sar, *args, **kwargs)

    return train, test


class PairedSARRGNIR(chainer.dataset.DatasetMixin):
    def __init__(self, rgnir, sar, *args, **kwargs):
        self.rgnir = rgnir
        self.sar = sar
        self.size = kwargs.pop('size')
        self.augmentation = kwargs.pop('augmentation')

    def __len__(self):
        return len(self.rgnir)

    def transform(self, sar, rgnir):
        c, h, w = rgnir.shape
        if self.augmentation:
            top = random.randint(0, h - self.size - 1)
            left = random.randint(0, w - self.size - 1)
            if random.randint(0, 1):
                rgnir = rgnir[:, :, ::-1]
                sar = sar[:, :, ::-1]
        else:
            top = (h - self.size) // 2
            left = (w - self.size) // 2
        bottom = top + self.size
        right = left + self.size

        sar = sar[:, top:bottom, left:right]
        rgnir = rgnir[:, top:bottom, left:right]

        sar = sar / 128. - 1.
        rgnir = rgnir / 128. - 1.

        return sar, rgnir

    def get_example(self, i):
        sar = cv2.imread(self.sar[i], 0)
        sar = sar[None, ...]
        rgnir = cv2.imread(self.rgnir[i], -1)
        rgnir = rgnir.transpose(2, 0, 1)
        sar, rgnir = self.transform(sar, rgnir)
        return sar, rgnir
