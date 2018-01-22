#!/usr/bin/env python
"""Test a pretrained CNN for Google speech commands."""

__author__ = 'Yuan Xu, Erdene-Ochir Tuguldur'

import argparse
import time
import csv
import os

from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import *
import torchnet

from datasets import *
from transforms import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset-dir", type=str, default='datasets/speech_commands/test', help='path of test dataset')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=3, help='number of workers for dataloader')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--multi-crop', action='store_true', help='apply crop and average the results')
parser.add_argument('--generate-kaggle-submission', action='store_true', help='generate kaggle submission file')
parser.add_argument("--kaggle-dataset-dir", type=str, default='datasets/speech_commands/kaggle', help='path of kaggle test dataset')
parser.add_argument('--output', type=str, default='', help='save output to file for the kaggle competition, if empty the model name will be used')
#parser.add_argument('--prob-output', type=str, help='save probabilities to file', default='probabilities.json')
parser.add_argument("model", help='a pretrained neural network model')
args = parser.parse_args()

dataset_dir = args.dataset_dir
if args.generate_kaggle_submission:
    dataset_dir = args.kaggle_dataset_dir

print("loading model...")
model = torch.load(args.model)
model.float()

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True
    model.cuda()

n_mels = 32
if args.input == 'mel40':
    n_mels = 40

feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
transform = Compose([LoadAudio(), FixAudioLength(), feature_transform])
test_dataset = SpeechCommandsDataset(dataset_dir, transform, silence_percentage=0)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=None,
                            pin_memory=use_gpu, num_workers=args.dataload_workers_nums)

criterion = torch.nn.CrossEntropyLoss()

def multi_crop(inputs):
    b = 1
    size = inputs.size(3) - b * 2
    patches = [inputs[:, :, :, i*b:size+i*b] for i in range(3)]
    outputs = torch.stack(patches)
    outputs = outputs.view(-1, inputs.size(1), inputs.size(2), size)
    outputs = torch.nn.functional.pad(outputs, (b, b, 0, 0), mode='replicate')
    return torch.cat((inputs, outputs.data))

def test():
    model.eval()  # Set model to evaluate mode

    #running_loss = 0.0
    #it = 0
    correct = 0
    total = 0
    confusion_matrix = torchnet.meter.ConfusionMeter(len(CLASSES))
    predictions = {}
    probabilities = {}

    pbar = tqdm(test_dataloader, unit="audios", unit_scale=test_dataloader.batch_size)
    for batch in pbar:
        inputs = batch['input']
        inputs = torch.unsqueeze(inputs, 1)
        targets = batch['target']

        n = inputs.size(0)
        if args.multi_crop:
            inputs = multi_crop(inputs)

        inputs = Variable(inputs, volatile = True)
        targets = Variable(targets, requires_grad=False)

        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda(async=True)

        # forward
        outputs = model(inputs)
        #loss = criterion(outputs, targets)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        if args.multi_crop:
            outputs = outputs.view(-1, n, outputs.size(1))
            outputs = torch.mean(outputs, dim=0)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

        # statistics
        #it += 1
        #running_loss += loss.data[0]
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).sum()
        total += targets.size(0)
        confusion_matrix.add(pred, targets.data)

        filenames = batch['path']
        for j in range(len(pred)):
            fn = filenames[j]
            predictions[fn] = pred[j][0]
            probabilities[fn] = outputs.data[j].tolist()

    accuracy = correct/total
    #epoch_loss = running_loss / it
    print("accuracy: %f%%" % (100*accuracy))
    print("confusion matrix:")
    print(confusion_matrix.value())

    return probabilities, predictions

print("testing...")
probabilities, predictions = test()
if args.generate_kaggle_submission:
    output_file_name = "%s" % os.path.splitext(os.path.basename(args.model))[0]
    if args.multi_crop:
        output_file_name = "%s-crop" % output_file_name
    output_file_name = "%s.csv" % output_file_name
    if args.output:
        output_file_name = args.output
    print("generating kaggle submission file '%s'..." % output_file_name)
    with open(output_file_name, 'w') as outfile:
        fieldnames = ['fname', 'label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for fname, pred in predictions.items():
            writer.writerow({'fname': os.path.basename(fname), 'label': test_dataset.classes[pred]})
