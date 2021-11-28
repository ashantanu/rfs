from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

from models import model_dict, model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from eval.meta_eval import meta_test_moco


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12_moco', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='data', metavar='N',
                        help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N',
                        help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    opt = parser.parse_args()

    if 'trainval' in opt.model_path:
        opt.use_trainval = True
    else:
        opt.use_trainval = False

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.data_root = '/data/vision/phillipi/rep-learn/{}'.format(opt.dataset)
        opt.data_aug = True
    elif hostname.startswith('instance'):
        opt.data_root = '/mnt/globalssd/fewshot/{}'.format(opt.dataset)
        opt.data_aug = True
    elif opt.data_root != 'data':
        opt.data_aug = True
    else:
        raise NotImplementedError('server invalid: {}'.format(hostname))

    return opt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    opt = parse_option()

    # test loader
    args = opt
    args.batch_size = args.test_batch_size
    # args.n_aug_support_samples = 1

    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        crop = 0.2
        image_size = 84
        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        crop_padding = 32
        normalize = transforms.Normalize(mean=mean, std=std)
        train_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(image_size, scale=(crop, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ])
        
        test_trans = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.Resize(image_size + crop_padding),
            transforms.CenterCrop(image_size),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize,
        ])
        
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans,
                                                        fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans,
                                                       fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans,
                                                  fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans,
                                                 fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    # load model
    model = create_model(opt.model, n_cls, opt.dataset)
    if torch.cuda.is_available():
        ckpt = torch.load(opt.model_path)
    else:
        ckpt = torch.load(opt.model_path,map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    print("Number of model params = ",count_parameters(model))

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # evalation
    start = time.time()
    val_acc, val_std = meta_test_moco(model, meta_valloader)
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std,
                                                                  val_time))

    start = time.time()
    val_acc_feat, val_std_feat = meta_test_moco(model, meta_valloader, use_logit=False)
    val_time = time.time() - start
    print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat,
                                                                       val_std_feat,
                                                                       val_time))

    start = time.time()
    test_acc, test_std = meta_test_moco(model, meta_testloader)
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test_moco(model, meta_testloader, use_logit=False)
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main()
