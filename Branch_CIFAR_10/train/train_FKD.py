import os
import sys
import math
import time
import shutil
import argparse
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('../')
from utils import AverageMeter, accuracy, get_parameters
from baseline import get_network as ti_get_network
from relabel.utils_fkd import ImageFolder_FKD_MIX, ComposeWithCoords, RandomResizedCropWithCoords, \
    RandomHorizontalFlipWithRes, mix_aug
import relabel.models as ti_models


# It is imported for you to access and modify the PyTorch source code (via Ctrl+Click), more details in README.md
from torch.utils.data._utils.fetch import _MapDatasetFetcher

import torch


def cosine_similarity(a, b, eps=1e-5):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-5):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DISTLoss(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, beta=2, gamma=2, tem=4):
        super(DISTLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tem = tem

    def forward(self, logits_student, logits_teacher):
        y_s = (logits_student / self.tem).softmax(dim=1)
        y_t = (logits_teacher / self.tem).softmax(dim=1)
        inter_loss = (self.tem ** 2) * inter_class_relation(y_s, y_t)
        intra_loss = (self.tem ** 2) * intra_class_relation(y_s, y_t)
        loss_kd = self.beta * inter_loss + self.gamma * intra_loss

        return loss_kd


def get_args():
    parser = argparse.ArgumentParser("FKD Training on CIFAR-10")
    parser.add_argument('--batch-size', type=int,
                        default=64, help='batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=1, help='gradient accumulation steps for small gpu memory')
    parser.add_argument('--start-epoch', type=int,
                        default=0, help='start epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='total epoch')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--loss-type', type=str, default="kl",
                        help='the type of the loss function')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='/path/to/cifar10', help='path to validation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='./save/1024', help='path to output dir')

    parser.add_argument('--cos', default=False,
                        action='store_true', help='cosine lr scheduler')
    parser.add_argument('--sgd', default=False,
                        action='store_true', help='sgd optimizer')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=0.064, help='sgd init learning rate')  # checked
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='sgd momentum')  # checked
    parser.add_argument('--weight-decay', type=float,
                        default=5e-4, help='sgd weight decay')  # checked
    parser.add_argument('--adamw-lr', type=float,
                        default=0.0005, help='adamw learning rate')
    parser.add_argument('--sgd-lr', type=float,
                        default=0.05, help='adamw learning rate')
    parser.add_argument('--adamw-weight-decay', type=float,
                        default=0.0, help='adamw weight decay')
    parser.add_argument('--ce-weight', type=float,
                        default=0.1, help='the weight og cross-entropy loss')

    parser.add_argument('--model', type=str,
                        default='ConvNetW128', help='student model name')
    parser.add_argument('-T', '--temperature', type=float,
                        default=3.0, help='temperature for distillation loss')
    parser.add_argument('--fkd-path', type=str,
                        default=None, help='path to fkd label')
    parser.add_argument('--wandb-project', type=str,
                        default='Temperature', help='wandb project name')
    parser.add_argument('--wandb-api-key', type=str,
                        default=None, help='wandb api key')
    parser.add_argument('--mix-type', default=None, type=str,
                        choices=['mixup', 'cutmix', None], help='mixup or cutmix or None')
    parser.add_argument('--fkd_seed', default=42, type=int,
                        help='seed for batch loading sampler')

    args = parser.parse_args()

    args.mode = 'fkd_load'
    return args


def main():
    args = get_args()

    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, name=args.output_dir.split('/')[-1])

    if not torch.cuda.is_available():
        raise Exception("need gpu to train!")

    assert os.path.exists(args.train_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Data loading
    normalize = transforms.Normalize([0.5071, 0.4867, 0.4408],
                                     [0.2675, 0.2565, 0.2761])
    train_dataset = ImageFolder_FKD_MIX(
        fkd_path=args.fkd_path,
        mode=args.mode,
        seed=args.fkd_seed,
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=args.train_dir,
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(size=32,
                                        scale=(0.08, 1),
                                        interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            normalize,
        ]))

    generator = torch.Generator()
    generator.manual_seed(args.fkd_seed)

    # only main process, no worker process
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
    #     num_workers=0, pin_memory=True,
    #     prefetch_factor=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # load validation data
    val_dataset = torchvision.datasets.CIFAR10(root=args.val_dir, train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   normalize,
                                               ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    print('load data successfully')

    # load student model
    print("=> loading student model '{}'".format(args.model))
    if "ConvNet" in args.model:
        model = ti_get_network(args.model, channel=3, num_classes=10, im_size=(32, 32), dist=False)
    else:
        model = ti_models.model_dict[args.model](num_classes=10)
    model = nn.DataParallel(model).cuda()
    model.train()

    if args.sgd:
        optimizer = torch.optim.SGD(get_parameters(model),
                                    lr=args.sgd_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(get_parameters(model),
                                      lr=args.adamw_lr,
                                      weight_decay=args.adamw_weight_decay)

    if args.cos == True:
        scheduler = LambdaLR(optimizer,
                             lambda step: 0.5 * (
                                     1. + math.cos(math.pi * step / args.epochs)) if step <= args.epochs else 0,
                             last_epoch=-1)
    else:
        scheduler = LambdaLR(optimizer,
                             lambda step: (1.0 - step / args.epochs) if step <= args.epochs else 0, last_epoch=-1)

    args.best_acc1 = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    for epoch in range(args.start_epoch, args.epochs):
        print(f"\nEpoch: {epoch}")

        global wandb_metrics
        wandb_metrics = {}

        train(model, args, epoch)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            top1 = validate(model, args, epoch)
        else:
            top1 = 0

        wandb.log(wandb_metrics)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = top1 > args.best_acc1
        args.best_acc1 = max(top1, args.best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': args.best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best, output_dir=args.output_dir)


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function_kl = nn.KLDivLoss(reduction='batchmean')
    loss_function_dist = DISTLoss(tem=args.temperature)

    model.train()
    t1 = time.time()
    args.train_loader.dataset.set_epoch(epoch)
    for batch_idx, batch_data in enumerate(args.train_loader):
        images, target, flip_status, coords_status = batch_data[:4]
        mix_index, mix_lam, mix_bbox, soft_label = batch_data[4:]

        images = images.cuda()
        target = target.cuda()
        soft_label = soft_label.cuda().float()  # convert to float32
        right = torch.all((mix_bbox[0].float() - mix_bbox[0].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[1].float() - mix_bbox[1].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[2].float() - mix_bbox[2].float().mean() + 1).bool()) \
                & torch.all((mix_bbox[3].float() - mix_bbox[3].float().mean() + 1).bool())
        if right.item() == 0:
            raise KeyError("In a batch, the mix_bbox should be same!")
        mix_bbox = [mix_bbox[0][0], mix_bbox[1][0], mix_bbox[2][0], mix_bbox[3][0]]
        images, _, _, _ = mix_aug(images, args, mix_index, mix_lam, mix_bbox)

        optimizer.zero_grad()
        assert args.batch_size % args.gradient_accumulation_steps == 0
        small_bs = args.batch_size // args.gradient_accumulation_steps

        # images.shape[0] is not equal to args.batch_size in the last batch, usually
        if batch_idx == len(args.train_loader) - 1:
            accum_step = math.ceil(images.shape[0] / small_bs)
        else:
            accum_step = args.gradient_accumulation_steps

        for accum_id in range(accum_step):
            partial_images = images[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_target = target[accum_id * small_bs: (accum_id + 1) * small_bs]
            partial_soft_label = soft_label[accum_id * small_bs: (accum_id + 1) * small_bs]

            
            output = model(partial_images)
            prec1, prec5 = accuracy(output, partial_target, topk=(1, 5))

            if args.loss_type == "kl":
                output = F.log_softmax(output / args.temperature, dim=1)
                partial_soft_label = F.softmax(partial_soft_label / args.temperature, dim=1)
                loss = loss_function_kl(output, partial_soft_label)
            elif args.loss_type == "dist":
                loss = loss_function_dist(output, partial_soft_label)
            elif args.loss_type == "mse_gt":
                loss = F.mse_loss(output, partial_soft_label) + F.cross_entropy(output, partial_target) * args.ce_weight
            else:
                raise NotImplementedError
            # loss = loss * args.temperature * args.temperature
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            n = partial_images.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()

    metrics = {
        "train/loss": objs.avg,
        "train/Top1": top1.avg,
        "train/Top5": top5.avg,
        "train/lr": scheduler.get_last_lr()[0],
        "train/epoch": epoch, }
    wandb_metrics.update(metrics)
    
    printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_last_lr()[0], objs.avg) + \
                'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
                'train_time = {:.6f}'.format((time.time() - t1))
    print(printInfo)

    with open(f"{args.model}_{args.ce_weight}_log.txt", 'w') as file:
        file.write(f"{printInfo}\n")
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
              'Top-5 err = {:.6f},\t'.format(100 - top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    print(logInfo)

    metrics = {
        'val/loss': objs.avg,
        'val/top1': top1.avg,
        'val/top5': top5.avg,
        'val/epoch': epoch,
    }
    wandb_metrics.update(metrics)

    return top1.avg


def save_checkpoint(state, is_best, output_dir=None, epoch=None):
    if epoch is None:
        path = output_dir + '/' + 'checkpoint.pth.tar'
    else:
        path = output_dir + f'/checkpoint_{epoch}.pth.tar'
    torch.save(state, path)

    if is_best:
        path_best = output_dir + '/' + 'model_best.pth.tar'
        shutil.copyfile(path, path_best)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method('spawn')
    main()
    wandb.finish()
