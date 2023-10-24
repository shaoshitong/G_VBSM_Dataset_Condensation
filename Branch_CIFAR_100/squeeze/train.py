import os
import argparse
import sys 
sys.path.append("../")
import torch
import torchvision
from torchvision import transforms
import copy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from baseline import get_network as ti_get_network
from models import model_dict

class KDLoss(nn.KLDivLoss):
    """
    "Distilling the Knowledge in a Neural Network"
    """
    def __init__(self, temperature=4, reduction='batchmean', **kwargs):
        super().__init__(reduction=reduction)
        self.temperature = temperature
        self.cel_reduction = 'mean' if reduction == 'batchmean' else reduction


    def forward(self, student_output, teacher_output, *args, **kwargs):
        soft_loss = super().forward(torch.log_softmax(student_output / self.temperature, dim=1),
                                    torch.softmax(teacher_output / self.temperature, dim=1))
        return (self.temperature ** 2) * soft_loss

class EMAMODEL(object):
    def __init__(self,model):
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)
        self.ema_model.eval()

    @torch.no_grad()
    def ema_step(self,model=None,decay_rate=0.999):
        for param, ema_param in zip(model.parameters(),self.ema_model.parameters()):
            ema_param.data.mul_(decay_rate).add_(param.data, alpha=1. - decay_rate)
    
    @torch.no_grad()
    def ema_swap(self,model=None):
        for param,ema_param in zip(self.ema_model.parameters(),model.parameters()):
            tmp = param.data.detach()
            param.data = ema_param.detach()
            ema_param.data = tmp
    
    @torch.no_grad()
    def __call__(self,x):
        return self.ema_model(x)

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == "CIFAR-100":
         trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))
    elif args.dataset == "CIFAR-10":
         trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    assert args.model in ["ResNet18", "MobileNetV2", "ShuffleNetV2_0_5", "WRN_16_2", "ConvNetW128"], f"{args.model} must be one of ResNet18, MobileNetV2, ShuffleNetV2_0_5, WRN_16_2!"
    if args.model == "ConvNetW128":
        model = ti_get_network(args.model , channel = 3, num_classes = 100, im_size = (32, 32), dist = False)
    else:
        model = model_dict[args.model](num_classes = 100 if args.dataset == "CIFAR-100" else 10)
    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.squeeze_path, args.dataset, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ''' organize the real dataset '''

    criterion = nn.CrossEntropyLoss().to(args.device)
    ema_criterion = KDLoss().to(args.device)
    model = model.to(args.device)
    model.train()
    lr = args.lr_teacher
    ema_model = EMAMODEL(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)

    for e in range(args.train_epochs):
        total_acc = 0
        total_number = 0
        model.train()

        for batch_idx, (input, target) in enumerate(trainloader):
            input = input.float().cuda()
            target = target.cuda()
            target = target.view(-1)
            optimizer.zero_grad()
            logit = model(input)
            ema_logit = ema_model(input)
            loss = criterion(logit, target) # + 0.4 * ema_criterion(logit,ema_logit)
            loss.backward()
            optimizer.step()
            ema_model.ema_step(model)

            total_acc += (target == logit.argmax(1)).float().sum().item()
            total_number += target.shape[0]
    
        top_1_acc = round(total_acc * 100 / total_number,3)
        print(f"Epoch: {e}, Top-1 Accuracy: {top_1_acc}%")
        if e % 10 == 0:
            state_dict = model.state_dict()
            torch.save(state_dict,os.path.join(save_dir,f"squeeze_{args.model}.pth"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR-100', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--lr_teacher', type=float, default=0.05, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--squeeze_path', type=str, default='./squeeze_wo_ema/', help='squeeze path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    args = parser.parse_args()
    main(args)

