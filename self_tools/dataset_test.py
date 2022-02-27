import argparse
# import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import over_write_args_from_file
# from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup
# from models.flexmatch.flexmatch import FlexMatch
from datasets.ssl_dataset import SSL_Dataset
# from datasets.data_utils import get_data_loader








def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')








parser = argparse.ArgumentParser(description='')

'''
Saving & loading of the model.
'''
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('-sn', '--save_name', type=str, default='flexmatch')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('-o', '--overwrite', action='store_true')
parser.add_argument('--use_tensorboard', action='store_true', help='Use tensorboard to plot and save curves, otherwise save the curves locally.')

'''
Training Configuration of flexmatch
'''

parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
                    help='total number of training iterations')
parser.add_argument('--num_eval_iter', type=int, default=5000,
                    help='evaluation frequency')
parser.add_argument('-nl', '--num_labels', type=int, default=40)
parser.add_argument('-bsz', '--batch_size', type=int, default=64)
parser.add_argument('--uratio', type=int, default=7,
                    help='the ratio of unlabeled data to labeld data in each mini-batch')
parser.add_argument('--eval_batch_size', type=int, default=1024,
                    help='batch size of evaluation data loader (it does not affect the accuracy)')

parser.add_argument('--hard_label', type=str2bool, default=True)
parser.add_argument('--T', type=float, default=0.5)
parser.add_argument('--p_cutoff', type=float, default=0.95)
parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
parser.add_argument('--use_DA', type=str2bool, default=False)
parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=True)

'''
Optimizer configurations
'''
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=3e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
parser.add_argument('--clip', type=float, default=0)
'''
Backbone Net Configurations
'''
parser.add_argument('--net', type=str, default='WideResNet')
parser.add_argument('--net_from_name', type=str2bool, default=False)
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--widen_factor', type=int, default=2)
parser.add_argument('--leaky_slope', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.0)

'''
Data Configurations
'''

parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
parser.add_argument('--train_sampler', type=str, default='RandomSampler')
parser.add_argument('-nc', '--num_classes', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=1)

'''
multi-GPUs & Distrbitued Training
'''

## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='**node rank** for distributed training')
parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:10601', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', type=str2bool, default=True,
                    help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
# config file
parser.add_argument('--c', type=str, default='')

args = parser.parse_args()

over_write_args_from_file(args, args.c)

print(args.dataset)
# train_dset = SSL_Dataset(args, alg='flexmatch', name=args.dataset, train=True,
#                                 num_classes=args.num_classes, data_dir=args.data_dir)


lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)
print(type(lb_dset), type(ulb_dset))
