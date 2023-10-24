import os
import random
import warnings
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.nn as nn
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import InterpolationMode

import models as ti_models
from baseline import get_network as ti_get_network

from tiny_in_dataset import normalize, get_tinyimagenet_dataloaders
from tqdm import tqdm
from utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, RandomHorizontalFlipWithRes, \
    mix_aug

parser = argparse.ArgumentParser(description='FKD Soft Label Generation on Tiny-ImageNet w/ Mix Augmentation')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--temperature', default=20, type=float,
                    help='the temperature of FKD')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# FKD soft label generation args
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--input-size', default=64, type=int, metavar='S',
                    help='argument in RandomResizedCrop')
parser.add_argument("--min-scale-crops", type=float, default=0.08,
                    help="argument in RandomResizedCrop")
parser.add_argument("--max-scale-crops", type=float, default=1.,
                    help="argument in RandomResizedCrop")
parser.add_argument('--fkd-path', default='./FKD_soft_label',
                    type=str, help='path to save soft labels')
parser.add_argument('--use-fp16', dest='use_fp16', action='store_true',
                    help='save soft labels as `fp16`')
parser.add_argument('--mode', default='fkd_save', type=str, metavar='N', )
parser.add_argument('--fkd-seed', default=42, type=int, metavar='N')
parser.add_argument('--candidate-number', default=4, type=int)
parser.add_argument('--mix-type', default=None, type=str, choices=['mixup', 'cutmix', None],
                    help='mixup or cutmix or None')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--pre-train-path', type=str,
                    default='../squeeze/squeeze_wo_ema/',
                    help='where to load the pre-trained backbone')


# sharing_strategy = "file_system"
# torch.multiprocessing.set_sharing_strategy(sharing_strategy)
#
#
# def set_worker_sharing_strategy(worker_id: int) -> None:
#     torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main():
    args = parser.parse_args()

    if not os.path.exists(args.fkd_path):
        os.makedirs(args.fkd_path, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    aux_teacher = ["ResNet18", "ConvNetW128" ,"WRN_16_2", "MobileNetV2", "ShuffleNetV2_0_5"][:args.candidate_number]
    print("=> using pytorch pre-trained model '{}'".format(aux_teacher))
    model_teacher = []
    for name in aux_teacher:
        if name == "ConvNetW128":
            model = ti_get_network(name, channel=3, num_classes=200, im_size=(64, 64), dist=False)
        else:
            model = ti_models.model_dict[name](num_classes=200)
        model_teacher.append(model)
        checkpoint = torch.load(
            os.path.join(args.pre_train_path, "Tiny-ImageNet", name, f"squeeze_{name}.pth"),
            map_location="cpu")
        model_teacher[-1].load_state_dict(checkpoint)
    model = model_teacher

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            for _model in model:
                _model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                _model = torch.nn.parallel.DistributedDataParallel(_model, device_ids=[args.gpu])
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            for _model in model:
                _model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                _model = torch.nn.parallel.DistributedDataParallel(_model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        for _model in model:
            _model = _model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        for _model in model:
            _model = _model.cuda(gpu)

    # freeze all layers
    for _model in model:
        for name, param in _model.named_parameters():
            param.requires_grad = False

    cudnn.benchmark = True

    print("process data from {}".format(args.data))

    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        root=args.data,
        seed=args.fkd_seed,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=args.input_size,
                                        scale=(args.min_scale_crops,
                                               args.max_scale_crops),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True)

    for epoch in tqdm(range(args.epochs)):
        dir_path = os.path.join(args.fkd_path, 'epoch_{}'.format(epoch))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        save(train_loader, model, dir_path, args)
        # exit()


def validate(input, target):
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = input
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


@torch.no_grad()
def save(train_loader, model, dir_path, args):
    """Generate soft labels and save"""
    for _model in model:
        _model.eval()
    total_acc = 0.
    for batch_idx, (images, target, flip_status, coords_status, index) in enumerate(train_loader):
        # print(images.shape) # [batch_size, 3, 32, 32]
        # print(flip_status.shape) # [batch_size,]
        # print(coords_status.shape) # [batch_size, 4]
        images = images.cuda()
        split_point = int(images.shape[0] // 2)
        origin_images = images
        images, mix_index, mix_lam, mix_bbox = mix_aug(images, args)

        total_output = []
        for _model in model:
            cat_output = []
            output = _model(origin_images[:split_point])
            cat_output.append(output)
            output = _model(origin_images[split_point:])
            cat_output.append(output)
            output = torch.cat(cat_output, 0)
            total_output.append(output)

        output = torch.stack(total_output, 0)
        # acc = (output[-2].argmax(1) == target.to(output.device)).float().sum() / output[-2].shape[0]
        # total_acc += acc
        # norm = torch.norm(output, dim=[1,2], keepdim=True)
        # output = output / norm * norm.mean(0,keepdim=True)
        output = output.mean(0)
        acc = (output.argmax(1) == target.to(output.device)).float().sum() / output.shape[0]
        total_acc += acc
        if args.use_fp16:
            output = output.half()
        batch_config = [coords_status, flip_status, mix_index, mix_lam, mix_bbox, output.cpu(), index]
        batch_config_path = os.path.join(dir_path, 'batch_{}.tar'.format(batch_idx))
        torch.save(batch_config, batch_config_path)
    print("Top 1-Acc.:", round(total_acc.item() / len(train_loader) * 100, 3))


if __name__ == '__main__':
    main()
