# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse  # Parser for command-line options, arguments and sub-commands
import datetime  # module supplies classes for manipulating dates and times
import json  # exposes an API familiar to users of the standard library marshal and pickle modules. # API --> application programming interface
import numpy as np
import os
import time  # time access and conversions, provides various time-related functions
from pathlib import Path  # module offers classes representing filesystem paths with semantics appropiate for different operating systems

import torch
import torch.backends.cudnn as cudnn   # GPU-accelerated library of primitives for deep neural networks
from torch.utils.tensorboard import SummaryWriter   # provides a high-level API to create an event file in a given directory and add
                                                    # summaries and events to it
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm  # collection of image models, layers, utilities, optimizers, schedulers, data-loaders/ augmentations and reference training

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_  # do check what this one does

import models.util.misc as misc  # import misc.py ----> Run OK
from models.util.pos_embed import interpolate_pos_embed  # import pos_embed.py ----> Run OK
from models.util.misc import NativeScalerWithGradNormCount as NativeScaler  # import another stuff from misc (do check this in misc)
from models.util.lars import LARS   # import lars.py  ---> run OK
from models.util.crop import RandomResizedCrop  # import crop.py ---- run OK

import models.models_vit as models_vit  # import models_vit.py ----> run OK

from models.engine_finetune_EU import train_one_epoch, evaluate  # import engine_finetune_EU ----> run OK

from datasets.eurosat_dataset import EurosatDataset, Subset  # run ok and load the EuroSat data
from sklearn.model_selection import train_test_split  # split the data, split arrays or matrices into random train and test subsets
from cvtorchvision import cvtransforms  # some sort of mix between open cv ans torchvision
from sklearn.metrics import average_precision_score

# An argument is a value that is passed to a function when it is called. It might be a variable, value or object passed to a function or method as input
def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False) # container to whole the arguments # create the parser
    parser.add_argument('--batch_size', default=512, type=int, # defines batch size, see help for a description
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)  # defines the epochs
    parser.add_argument('--accum_iter', default=1, type=int,  # defines the iterations, see help, it better describes it
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')  # yep, this description is accurate

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)') # define weight decay

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')  # define learning rate
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR', # defines absolute learning rate
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0') # defines learning rate for cyclic schedulers

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR') # yep, epochs to warm up learning rate

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint') # defines finetuning from checkpoint
    parser.add_argument('--global_pool', action='store_true')  # maxpool stuff?
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification') # seems so, do check this part for #
                                                                               # classification, can be useful for different features prediction

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path') # yep data set path, it's like the 'root' for the dataloader mae pre-training code
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types') # definition of number of classes

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')  # as in the description
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')     # as in the description
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')   # as in the description
    parser.add_argument('--seed', default=0, type=int)  # define the seed for the GPU
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')   # as in the description

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')  # parser for epoch starting
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')  # parser for evaluation
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor') # description
    parser.add_argument('--num_workers', default=10, type=int) # num workers for GPU usage and training
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.') # description
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')  # ask about this
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # as in the description
    parser.add_argument('--local_rank', default=-1, type=int)  # ask about this
    parser.add_argument('--dist_on_itp', action='store_true')  # ask about this
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training') # as in the description
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')  # as in the description

    parser.add_argument("--is_slurm_job", action='store_true', help="slurm job") # defines a cluster management and job scheduling system
                                                                           # for large and small Linux clusters
    parser.add_argument("--train_frac", default=1.0, type=float, help="use a subset of labeled data") # as in the description

    return parser


def main(args):
    misc.init_distributed_mode(args)  # taking the 'parameters' previously defined in get_args_parser

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__)))) # path to redirect the trained data
    print("{}".format(args).replace(', ', ',\n'))  # replace when getting new data

    device = torch.device(args.device)  # object that represents the data type of a torch.Tensor

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank() # seed is mainly used for generating random numbers, used to initialize the random number generator
    torch.manual_seed(seed) # will set the seed of the random number generator to a fixed value
    np.random.seed(seed)

    cudnn.benchmark = True # enables benchmark mode in cudnn
                           # benchmark mode is good whenever your input sizes for your network do not vary

    train_transforms = cvtransforms.Compose([ # chaining together images to be transformed and augmented
        cvtransforms.RandomResizedCrop(224), # images variation (augmentation)
        cvtransforms.RandomHorizontalFlip(), # images variation
        cvtransforms.ToTensor(), # converting images to a tensor
    ])

    val_transforms = cvtransforms.Compose([  # same as described above
        cvtransforms.Resize(256),
        cvtransforms.CenterCrop(224),
        cvtransforms.ToTensor(),
    ])

    eurosat_dataset = EurosatDataset(root=args.data_path, normalize=False)  # data_path may be configured as teh macros with the folder I have the data

    indices = np.arange(len(eurosat_dataset))  # indices are now the lenght of the dataset
    train_indices, test_indices = train_test_split(indices, train_size=0.8, stratify=eurosat_dataset.targets,
                                                   random_state=args.seed)

    dataset_train = Subset(eurosat_dataset, train_indices, train_transforms) # setting up to start training dataset
    dataset_val = Subset(eurosat_dataset, test_indices, val_transforms)  # same but for validation

    if args.train_frac is not None and args.train_frac < 1:  # train frac was defined in the args parse section
        frac_indices = np.arange(len(dataset_train))
        sub_train_indices, sub_test_indices = train_test_split(frac_indices, train_size=args.train_frac, # here it is splitting the data but
                                                               random_state=args.seed)              # how to call train_frac?
        dataset_train = Subset(dataset_train, sub_train_indices)

    if True:  # args.distributed:
        num_tasks = args.world_size
        print(misc.get_world_size())
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        in_chans=13
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * args.world_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model_without_ddp.head.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=0)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, criterion)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch % 5 == 0 or (epoch + 1 == args.epochs):
            test_stats = evaluate(data_loader_val, model, device, criterion)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)