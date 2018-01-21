#!/usr/bin/env python
"""Test a pretrained CNN for CIFAR10."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import *
import torchnet

import models

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-root", type=str, default='./datasets', help='path of train dataset')
parser.add_argument("--test-batch-size", type=int, default=100, help='test batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=2, help='number of workers for dataloader')
parser.add_argument("model", help='a pretrained neural network model')
args = parser.parse_args()

print("loading model...")
model = torch.load(args.model)
model.float()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True
    model.cuda()

to_tensor_and_normalize = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False, download=True, transform=to_tensor_and_normalize)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.dataload_workers_nums)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = torch.nn.CrossEntropyLoss()

def test():
    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    it = 0
    correct = 0
    total = 0
    confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))

    pbar = tqdm(test_dataloader, unit="images", unit_scale=test_dataloader.batch_size)
    for batch in pbar:
        inputs, targets = batch
        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # statistics
        running_loss += loss.data[0]
        it += 1
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(pred, targets.data)

        # update the progress bar
        pbar.set_postfix({
            'loss': "%.05f" % (running_loss / it),
            'acc': "%.02f%%" % (100*correct/total)
        })

    accuracy = correct/total
    epoch_loss = running_loss / it
    print("accuracy: %f%%, loss: %f" % (100*accuracy, epoch_loss))
    print("confusion matrix:")
    print(confusion_matrix.value())

print("testing...")
test()
