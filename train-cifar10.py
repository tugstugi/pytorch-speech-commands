#!/usr/bin/env python
"""Train a CNN for CIFAR10."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time

from tqdm import *

import torch
from torch.autograd import Variable

import torchvision
from torchvision.transforms import *

import models

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--dataset-root", type=str, default='./datasets', help='path of train dataset')
parser.add_argument("--train-batch-size", type=int, default=128, help='train batch size')
parser.add_argument("--test-batch-size", type=int, default=100, help='test batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=2, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=5e-4, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=0.1, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=2, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=2, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=50, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=['vgg19_bn', 'wideresnet28_10', 'wideresnet28_10D', 'wideresnet52_10'], default='vgg19_bn', help='model of NN')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

to_tensor_and_normalize = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=True, download=True,
            transform=Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                to_tensor_and_normalize
            ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataload_workers_nums)

testset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False, download=True, transform=to_tensor_and_normalize)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.dataload_workers_nums)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# a name used to save checkpoints etc.
full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (args.model, args.optim, args.lr_scheduler, args.train_batch_size, args.learning_rate, args.weight_decay)
if args.comment:
    full_name = '%s_%s' % (full_name, args.comment)

if args.model == "wideresnet28_10":
    model = models.WideResNet(depth=28, widen_factor=10, dropRate=0, num_classes=len(CLASSES), in_channels=1)
if args.model == "wideresnet28_10D":
    model = models.WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=len(CLASSES), in_channels=1)
if args.model == "wideresnet52_10":
    model = models.WideResNet(depth=52, widen_factor=10, dropRate=0, num_classes=len(CLASSES), in_channels=1)
else:
    model = models.vgg19_bn(num_classes=len(CLASSES), in_channels=3)

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

criterion = torch.nn.CrossEntropyLoss()

if args.optim == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

start_epoch = 0
best_accuracy = 0

if args.resume:
    print("resuming a checkpoint '%s'" % args.resume)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model.float()
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_accuracy = checkpoint.get('accuracy', best_accuracy)
    #best_loss = checkpoint.get('loss', best_loss)
    start_epoch = checkpoint.get('epoch', start_epoch)

    del checkpoint  # reduce memory

if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

def get_lr():
    return optimizer.param_groups[0]['lr']

def train(epoch):
    print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
    model.train()  # Set model to training mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(trainloader, unit="images", unit_scale=trainloader.batch_size)
    for batch in pbar:
        inputs, targets = batch
        inputs = Variable(inputs, requires_grad=True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward/backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.data[0]
        it += 1
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    epoch_loss = running_loss / it

def test(epoch):
    global best_accuracy

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0

    pbar = tqdm(testloader, unit="images", unit_scale=testloader.batch_size)
    for batch in pbar:
        inputs, targets = batch
        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        running_loss += loss.data[0]
        it += 1
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        #'loss': epoch_loss,
        'accuracy': accuracy,
        'optimizer' : optimizer.state_dict(),
    }

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(checkpoint, 'checkpoints/best-cifar10-checkpoint-%s.pth' % full_name)
        torch.save(model, 'best-cifar10-model-%s.pth' % full_name)

    torch.save(checkpoint, 'checkpoints/last-cifar10-checkpoint.pth')
    del checkpoint  # reduce memory

    return epoch_loss

print("training...")
since = time.time()
for epoch in range(start_epoch, args.max_epochs):
    if args.lr_scheduler == 'step':
        lr_scheduler.step()

    train(epoch)
    epoch_loss = test(epoch)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=epoch_loss)

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60, time_elapsed % 60)
    print("%s, best test accuracy: %.02f%%" % (time_str, 100*best_accuracy))
print("finished")
