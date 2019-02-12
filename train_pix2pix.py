import os, sys
import shutil
import yaml
import matplotlib
matplotlib.use('Agg')

import argparse
import chainer
from chainer import training
from chainer.training import extensions
import multiprocessing

sys.path.append(os.path.dirname(__file__))

import source.yaml_utils as yaml_utils


def create_result_dir(result_dir, config_path, config):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(config_path, result_dir)
    copy_to_result_dir(
        config.models['generator']['fn'], result_dir)
    copy_to_result_dir(
        config.models['discriminator']['fn'], result_dir)
    copy_to_result_dir(
        config.dataset['fn'], result_dir)
    copy_to_result_dir(
        config.updater['fn'], result_dir)


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    dis_conf = config.models['discriminator']
    dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])
    return gen, dis


def make_optimizer(model, alpha=0.0002, beta1=0., beta2=0.9):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml', help='path to config file')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='directory to save the results to')
    parser.add_argument('--snapshot', type=str, default='',
                        help='path to the snapshot')
    parser.add_argument('--loaderjob', type=int,
                        help='number of parallel data loading processes')
    args = parser.parse_args()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    device = 0
    chainer.cuda.get_device_from_id(device).use()
    print("init")
    multiprocessing.set_start_method('forkserver')

    # Model
    gen, dis = load_models(config)
    gen.to_gpu()
    dis.to_gpu()
    models = {"gen": gen, "dis": dis}

    # Optimizer
    opt_gen = make_optimizer(gen, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opt_dis = make_optimizer(dis, alpha=config.adam['alpha'], beta1=config.adam['beta1'], beta2=config.adam['beta2'])
    opts = {"opt_gen": opt_gen, "opt_dis": opt_dis}

    # Dataset
    train, test = yaml_utils.load_dataset(config)

    # Iterator
    train_iter = chainer.iterators.MultiprocessIterator(train, config.batchsize, n_processes=args.loaderjob)
    test_iter = chainer.iterators.SerialIterator(test, config.batchsize_test, repeat=False, shuffle=False)

    kwargs = config.updater['args'] if 'args' in config.updater else {}
    kwargs.update({
        'models': models,
        'iterator': train_iter,
        'optimizer': opts,
        'device': device,
    })
    updater = yaml_utils.load_updater_class(config)
    updater = updater(**kwargs)
    out = args.results_dir
    create_result_dir(out, args.config_path, config)
    trainer = training.Trainer(updater, (config.iteration, 'iteration'), out=out)
    report_keys = ["loss_dis", "loss_gen", "loss_l1", "psnr", "ssim"]
    eval_func = yaml_utils.load_eval_func(config)
        # Set up logging
    trainer.extend(extensions.snapshot(), trigger=(config.snapshot_interval, 'iteration'))
    for m in models.values():
        trainer.extend(extensions.snapshot_object(
            m, m.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
    trainer.extend(eval_func(test_iter, gen, dst=out), trigger=(config.evaluation_interval, 'iteration'), priority=chainer.training.extension.PRIORITY_WRITER)
    for key in report_keys:
        trainer.extend(extensions.PlotReport(key, trigger=(config.evaluation_interval, 'iteration'), file_name='{}.png'.format(key)))
    trainer.extend(extensions.ProgressBar(update_interval=config.progressbar_interval))
    ext_opt_gen = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_gen)
    ext_opt_dis = extensions.LinearShift('alpha', (config.adam['alpha'], 0.),
                                         (config.iteration_decay_start, config.iteration), opt_dis)
    trainer.extend(ext_opt_gen)
    trainer.extend(ext_opt_dis)
    if args.snapshot:
        print("Resume training with snapshot:{}".format(args.snapshot))
        chainer.serializers.load_npz(args.snapshot, trainer)

    # Run the training
    print("start training")
    trainer.run()


if __name__ == '__main__':
    main()
