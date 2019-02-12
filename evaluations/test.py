import sys
sys.path.append('..')
import os
import numpy as np
import argparse
import yaml
import cv2
from skimage.measure import compare_ssim, compare_psnr

import chainer

import source.yaml_utils as yaml_utils
from evaluations.evaluation import get_batch


def load_gen(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def save_images(input_images, output_images, gt_images, results_dir, current_n):
    psnr = []
    ssim = []
    for i, (x, y, t) in enumerate(zip(input_images, output_images, gt_images)):
        x = x.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_input_.png'.format(current_n + i)), x)
        y = y.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_output_.png'.format(current_n + i)), y)
        t = t.transpose(1, 2, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(results_dir, '{:03d}_gt.png'.format(current_n + i)), t)

        psnr += [compare_psnr(y, t)]
        ssim += [compare_ssim(y, t, multichannel=True)]

    return psnr, ssim


def test(args):
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    chainer.cuda.get_device_from_id(0).use()
    gen = load_gen(config)
    chainer.serializers.load_npz(args.gen_model, gen)
    gen.to_gpu()
    xp = gen.xp
    _, test = yaml_utils.load_dataset(config)
    test_iter = chainer.iterators.SerialIterator(test, config.batchsize_test, repeat=False, shuffle=False)

    results_dir = args.results_dir
    images_dir = os.path.join(results_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    n = 0
    psnr = []
    ssim = []
    while True:
        x, t, batchsize = get_batch(test_iter, xp)

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = gen(x)
            x = x.array.get() * 127.5 + 127.5
            y = np.clip(y.array.get() * 127.5 + 127.5, 0., 255.)
            t = t.array.get() * 127.5 + 127.5

            _psnr, _ssim = save_images(x, y, t, images_dir, current_n=n)
            psnr += _psnr
            ssim += _ssim
            n += len(x)

        if test_iter.is_new_epoch:
            test_iter.reset()
            break

    print('psnr: {}'.format(np.mean(psnr)))
    print('ssim: {}'.format(np.mean(ssim)))

    psnr = list(map(str, psnr))
    ssim = list(map(str, ssim))

    with open(os.path.join(results_dir, 'psnr.txt'), 'w') as f:
        f.write('\n'.join(psnr))
    with open(os.path.join(results_dir, 'ssim.txt'), 'w') as f:
        f.write('\n'.join(ssim))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--gen_model', type=str, required=True)
    args = parser.parse_args()

    test(args)
