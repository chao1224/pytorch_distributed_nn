import sys
import math
import threading
import argparse
import time
import collections
from itertools import repeat

import torch
from torch.autograd import Variable
from torch._utils import _flatten_tensors, _unflatten_tensors
from torch.cuda.comm import broadcast_coalesced
from torch.cuda import nccl
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from data_loader_ops.my_data_loader import DataLoader

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args

# we use LeNet here for our simple case
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # (in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, target):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        loss = self.criterion(x, target)
        return x, loss
    def name(self):
        return 'lenet'

class LeNetSplit(nn.Module):
    def __init__(self):
        super(LeNetSplit, self).__init__()
        self.layers0 = nn.ModuleList([
            nn.Conv2d(1, 20, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 50, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            ])
        self.layers1 = nn.ModuleList([
            nn.Linear(4*4*50, 500),
            nn.Linear(500, 10),
            ])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        self.output = []
        self.input = []
        for layer in self.layers0:
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # compute output
            x = layer(x)
            # add to list of outputs
            self.output.append(x)
        x = x.view(-1, 4*4*50)
        for layer in self.layers1:
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # compute output
            x = layer(x)
            # add to list of outputs
            self.output.append(x)
        return x

    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)

class LeNetLearner:
    """a deprecated class, please don't call this one in any time"""
    def __init__(self, **kwargs):
        self._step_changed = False
        self._update_step = False
        self._new_step_queued = 0

        self._cur_step = 0
        self._next_step = self._cur_step + 1
        self._step_fetch_request = False
        self.max_num_epochs = kwargs['max_epochs']
        self.lr = kwargs['learning_rate']
        self.momentum = kwargs['momentum']

    def build_model(self):
        #self.network = LeNet()
        self.network = LeNetSplit()

        # this is only used for test
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)

    def train(self, train_loader=None):
        self.network.train()

        # iterate of epochs
        for i in range(self.max_num_epochs):            
            for batch_idx, (data, y_batch) in enumerate(train_loader):
                iter_start_time = time.time()
                data, target = Variable(data, requires_grad=True), Variable(y_batch)
                self.optimizer.zero_grad()
                logits = self.network(data)
                logits_1 = Variable(logits.data, requires_grad=True)
                
                loss = self.network.criterion(logits_1, target)
                print("Trial Loss: {}".format(loss.data[0]))
                loss.backward()
                
                self.network.backward(logits_1.grad)

                self.optimizer.step()

                prec1, prec5 = accuracy(logits.data, y_batch, topk=(1, 5))
                # load the training info
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Prec@1: {}  Prec@5: {}  Time Cost: {}'.format(
                    i, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0], 
                    prec1.numpy()[0], 
                    prec5.numpy()[0], time.time()-iter_start_time))

    def update_state_dict(self):
        """for this test version, we set all params to zeros here"""
        # we need to build a state dict first
        new_state_dict = {}
        for key_name, param in self.network.state_dict().items():
            tmp_dict = {key_name: torch.FloatTensor(param.size()).zero_()}
            new_state_dict.update(tmp_dict)
        self.network.load_state_dict(new_state_dict)


if __name__ == "__main__":
	args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

	kwargs = {'batch_size':args.batch_size, 'learning_rate':args.lr, 'max_epochs':args.epochs, 'momentum':args.momentum}

	# load training and test set here:
	train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
	               transform=transforms.Compose([
	                   transforms.ToTensor(),
	                   transforms.Normalize((0.1307,), (0.3081,))
	               ])), batch_size=args.batch_size, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
	    datasets.MNIST('../data', train=False, transform=transforms.Compose([
	                       transforms.ToTensor(),
	                       transforms.Normalize((0.1307,), (0.3081,))
	                   ])), batch_size=args.test_batch_size, shuffle=True)

	nn_learner = LeNetLearner(**kwargs)
	nn_learner.build_model()
	nn_learner.train(train_loader=train_loader)