import os
import numpy as np
import cv2
from skimage.measure import compare_ssim, compare_psnr

import chainer
from chainer import Variable


def get_batch(test_iter, xp):
    batch = test_iter.next()
    batchsize = len(batch)

    x = []
    gt = []
    for j in range(batchsize):
        x.append(np.asarray(batch[j][0]).astype("f"))
        gt.append(np.asarray(batch[j][1]).astype("f"))
    x = Variable(xp.asarray(x))
    gt = Variable(xp.asarray(gt))

    return x, gt, batchsize


def save_images(input_image, output_image, gt_image, results_dir, current_n):
    psnr = 0
    ssim = 0
    for i, (x, y, t) in enumerate(zip(input_image, output_image, gt_image)):
        x = x.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_input.png'.format(current_n + i)), x)
        y = y.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_output.png'.format(current_n + i)), y)
        t = t.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_gt.png'.format(current_n + i)), t)

        psnr += compare_psnr(y, t)
        ssim += compare_ssim(y, t, multichannel=True)

    return psnr, ssim


def save_images_(input_image, output_image, gt_image, results_dir, current_n):
    psnr = 0
    ssim = 0
    for i, (x, y, t) in enumerate(zip(input_image, output_image, gt_image)):
        x = x.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_input.png'.format(current_n + i)), x)
        y = y.astype(np.uint8)
        for j, image in enumerate(y):
            cv2.imwrite(os.path.join(results_dir, '{:03d}_output_{}.png'.format(current_n + i, j)), image)
        t = t.astype(np.uint8)
        for j, image in enumerate(t):
            cv2.imwrite(os.path.join(results_dir, '{:03d}_gt_{}.png'.format(current_n + i, j)), image)

        psnr += compare_psnr(y, t)
        ssim += compare_ssim(y, t, multichannel=True)

    return psnr, ssim


def out_image(test_iter, gen, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        test_iter.reset()
        xp = gen.xp
        results_dir = os.path.join(dst, 'test_{:03d}'.format(trainer.updater.iteration))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        n = 0
        sum_psnr = 0
        sum_ssim = 0
        while True:
            x, gt, batchsize = get_batch(test_iter, xp)

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                out = gen(x)
                out = np.clip(out.array.get() * 127.5 + 127.5, 0., 255.)
                gt = gt.array.get() * 127.5 + 127.5
                x = x.array.get() * 127.5 + 127.5

                if out.shape[1] == 1 or out.shape[1] == 3:
                    _psnr, _ssim = save_images(x, out, gt, results_dir, current_n=n)
                else:
                    _psnr, _ssim = save_images_(x, out, gt, results_dir, current_n=n)
                sum_psnr += _psnr
                sum_ssim += _ssim

                n += len(out)

            if test_iter.is_new_epoch:
                test_iter.reset()
                break

        psnr = sum_psnr / n
        ssim = sum_ssim / n
        chainer.reporter.report({'psnr': psnr,
                                 'ssim': ssim})

    return make_image
