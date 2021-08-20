import os
import random
import time
import numpy as np
import argparse

from src.utils import build_optim, build_lrscheduler, LRSchedulerC, VisualDLC, build_transform, TwoCropTransform
from src.models.supcon import SupConLoss, SupConResNet, SupConModel
from tools.configsys import CfgNode as CN

import paddle
from paddle.vision.datasets import Cifar10


def get_args_parser():
    parser = argparse.ArgumentParser('Set PaddlePaddle cifar10 config', add_help=False)
    parser.add_argument('-y', '--yaml', default='yamls/resnet50_supcon.yml', type=str)
    return parser


def main(cfg):
    paddle.seed(cfg.COMMON.seed)
    random.seed(cfg.COMMON.seed)
    np.random.seed(cfg.COMMON.seed)
    print("*" * 40)

    net = SupConResNet(cfg.CLASSIFIER.name, cfg.CLASSIFIER.head, cfg.CLASSIFIER.feat_dim)
    # print(net)
    model = SupConModel(net)

    lrs = build_lrscheduler(cfg.SCHEDULER)
    clip = paddle.nn.ClipGradByNorm(clip_norm=1)
    optim = build_optim(cfg.OPTIMIZER, parameters=net.parameters(), learning_rate=lrs, grad_clip=clip)
    train_transforms, test_transforms = build_transform()
    train_set = Cifar10(cfg.COMMON.data_path, mode='train', transform=TwoCropTransform(train_transforms))
    test_set = Cifar10(cfg.COMMON.data_path, mode='test', transform=TwoCropTransform(test_transforms))
    vis_name = '/{}-{}-{}'.format(cfg.CLASSIFIER.name, cfg.CLASSIFIER.mode,
                                  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    log_dir = cfg.COMMON.logdir + vis_name
    callbacks = [LRSchedulerC(), VisualDLC(log_dir)]

    model.prepare(optim, loss=SupConLoss(temperature=cfg.COMMON.temp))

    # load checkpoint
    if "continue_from" in cfg.COMMON:
        print("Restore checkpoint from", cfg.COMMON.continue_from)
        model.load(cfg.COMMON.continue_from)

    model.fit(
        train_set,
        test_set,
        batch_size=cfg.COMMON.batch_size,
        epochs=cfg.COMMON.epochs,
        eval_freq=2,
        num_workers=cfg.COMMON.workers,
        save_dir=log_dir,
        save_freq=cfg.COMMON.save_freq,
        verbose=cfg.COMMON.verbose,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PaddlePaddle cifar10 classifier training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = CN.load_cfg(args.yaml)
    cfg.freeze()
    print(cfg)
    n_gpu = len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(","))
    print("num of GPUs:", n_gpu)
    if n_gpu > 1:
        paddle.distributed.spawn(main, args=(cfg,), nprocs=n_gpu)
    else:
        main(cfg)
