from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import subprocess

cwd = os.getcwd()
sys.path.append(cwd+'/../')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import models

from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from util import *
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange

class HAR_Dataset(Dataset):
    def __init__(self, split):
        root =  '/home/chia/.cache/UCI HAR Dataset/'
        sys.path.append(cwd + '/../../data/deep-learning-HAR/utils')
        from utilities import read_data
        self.imgs, self.labels, self.list_ch_train = read_data(data_path=root, split=split) # train
        # make sure they contain only valid labels [0 ~ class -1]
        self.labels = self.labels - 1

        assert len(self.imgs) == len(self.labels), "Mistmatch in length!"
        # Normalize?
        self.imgs = self.standardize(self.imgs)

    def standardize(self, data):
        """ Standardize data """
        # Standardize train and test
        standardized_data = (data - np.mean(data, axis=0)[None,:,:]) / np.std(data, axis=0)[None,:,:]
        return standardized_data

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        return len(self.imgs)

def save_state(model, acc):
    global logger_
    print('==> Saving model (accuracy {:.2f}) ....'.format(acc))
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    if (hasattr(model, 'weights_pruned')):
        state['weights_pruned'] = model.weights_pruned
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    subprocess.call('mkdir -p saved_models', shell=True)
    if args.prune:
        subprocess.call('mkdir -p saved_models/'+args.prune, shell=True)
        subprocess.call('mkdir -p saved_models/'+args.prune+'/'+args.arch, shell=True)
    if args.prune:
        torch.save(state, 'saved_models/'+args.prune+'/'+args.arch+'/'+'stage_'+str(args.stage)+'.pth.tar')
    else:
        torch.save(state, 'saved_models/'+args.arch+'.origin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.type(torch.float)), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.prune:
            prune_op.prune_weight()
        '''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''
    return

def my_train(model, optimizer, criterion, epoch, args, train_loader, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return

@torch.no_grad()
def test(evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    if args.prune:
        prune_op.prune_weight()
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.type(torch.float)), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(test_loader.dataset)
    if acc > best_acc:
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    #if args.prune == None or evaluate:
    test_loss /= len(test_loader.dataset)
    '''
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    '''
    return (test_loss * args.batch_size, acc, best_acc)

@torch.no_grad()
def my_test(model, args, test_loader, criterion, logger, evaluate=True):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset), 100. * acc))
    return test_loss * args.batch_size

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_epochs))
    # print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        if args.retrain and ('mask' in param_group['key']): # retraining
            param_group['lr'] = 0.0
        elif args.prune_target and ('mask' in param_group['key']):
            if args.prune_target in param_group['key']:
                param_group['lr'] = lr
            else:
                param_group['lr'] = 0.0
        else:
            param_group['lr'] = lr
    return lr

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
            help='number of epochs to train (default: 60)')
    parser.add_argument('--lr-epochs', type=int, default=15, metavar='N',
            help='number of epochs to decay the lr (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
            metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default=None,
            help='the MNIST network structure: LeNet_5_p | LeNet_5 | HAR | SqueezeNet')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
            help='whether to run evaluation')
    parser.add_argument('--retrain', action='store_true', default=False,
            help='retrain the pruned network')
    parser.add_argument('--prune', action='store', default=None,
            help='pruning mechanism: None | intermittent | energy')
    parser.add_argument('--prune_shape', action='store', default='vector',
            help='pruning shape: vector | channel')
    parser.add_argument('--prune-target', action='store', default=None,
            help='pruning target: default=None | conv | ip')
    parser.add_argument('--stage', action='store', type=int, default=0,
            help='pruning stage')
    parser.add_argument('--debug', action='store', type=int, default=-1,
            help='set debug level')
    parser.add_argument('--candidates-pruning-ratios', action='store', nargs='+', type=float, default=[0.25, 0.3, 0.35, 0.4],
            help='candidates of pruning ratios for weight pruning')
    parser.add_argument('--admm', action='store_true', default=False,
            help='w/ or w/o ADMM')
    parser.add_argument('--sa', action='store_true', default=False,
            help='w/ or w/o simulated annealling')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # check options
    if not (args.prune_target in [None, 'conv', 'ip']):
        print('ERROR: Please choose the correct prune_target')
        exit()

    print_args(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load data
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    # generate the model
    if args.arch == 'LeNet_5' or args.arch == 'LeNet_5_p':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        if args.arch == 'LeNet_5':
            model = models.LeNet_5(args.prune)
        else:
            model = models.LeNet_5_p(args.prune)
    elif args.arch == 'HAR':
        train_loader = torch.utils.data.DataLoader(
            HAR_Dataset(split='train'),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            HAR_Dataset(split='test'),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
        model = models.HAR_CNN(args.prune, n_channels=9, n_classes=6)
    elif args.arch == 'SqueezeNet':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])),
                batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)

        model = models.SqueezeNet(args.prune)
    else:
        print('ERROR: {} arch is not suppported'.format(args.arch))
        exit()

    if not args.pretrained:
        best_acc = 0.0
        model.weights_pruned = None
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        load_state(model, pretrained_model['state_dict'])
        if args.prune and ('weights_pruned' in pretrained_model.keys()):
            model.weights_pruned = pretrained_model['weights_pruned']
        else:
            model.weights_pruned = None

    if args.cuda:
        model.cuda()

    print(model)
    param_dict = dict(model.named_parameters())
    params = []


    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': args.lr,
            'momentum':args.momentum,
            'weight_decay': args.weight_decay,
            'key':key}]

    if args.arch == 'LeNet_5' or args.arch == 'LeNet_5_p':
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)
    elif args.arch == 'HAR':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.arch == 'SqueezeNet':
        optimizer = optim.Adam(params, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        print_layer_info(model)
        if args.prune:
            if not model.weights_pruned:
                raise Exception ('weights_pruned is missing')
        test(evaluate=True)
        exit()

    def evaluate_function(model, logger):
        return my_test(model, args, test_loader, criterion, logger)
    def trainer(model, optimizer, criterion, epoch, logger): #for ADMM pruning
        return my_train(model, optimizer, criterion, epoch, args, train_loader, logger)

    cur_epoch = 0
    cur_loss = 0
    cur_acc = 0
    best_acc = 0
    pbar = tqdm(iterable=range(1, args.epochs + 1), desc='[Epoch: {}| Loss: {:.4f}| Accuracy: {:.2f}| Best Accuracy: {:.2f}]'.format(cur_epoch, cur_loss, cur_acc, best_acc))
    if args.prune:
        admm_params = None
        print('==> Start pruning ...')
        if not args.pretrained:
            print('==> ERROR: Please assign the pretrained model')
            exit()
        if args.arch == 'LeNet_5' or args.arch == 'LeNet_5_p':
            input_shape = (1, 28, 28)
        elif args.arch == 'HAR':
            seq_len = 128
            n_channels = 9
            input_shape = (n_channels, seq_len)
        elif args.arch == 'SqueezeNet':
            input_shape = (3, 32, 32)

        if args.admm:
            admm_params = {
                'train_function': trainer,
                'optimizer': optimizer,
                'criterion': criterion
            }

        prune_op = Prune_Op(model, train_loader, criterion, input_shape, args, evaluate_function, admm_params=admm_params)
        if not args.admm:
            for epoch in pbar:
                if args.arch == 'LeNet_5' or args.arch == 'LeNet_5_p':
                    lr = adjust_learning_rate(optimizer, epoch)
                elif args.arch == 'SqueezeNet' or args.arch == 'HAR':
                    # adjusted by ADAM
                    pass
                train(epoch)
                cur_epoch = epoch
                cur_loss, cur_acc, best_acc = test()
                pbar.set_description('[Epoch: {}| Loss: {:.4f}| Accuracy: {:.2f}| Best Accuracy: {:.2f}]'.format(cur_epoch, cur_loss, cur_acc, best_acc))
        test(evaluate=True)
        # prune_op.print_info()
    else:
        for epoch in trange(1, args.epochs + 1):
            if args.arch == 'LeNet_5' or args.arch == 'LeNet_5_p':
                adjust_learning_rate(optimizer, epoch)
            elif args.arch == 'SqueezeNet' or args.arch == 'HAR':
                # adjusted by ADAM
                pass
            train(epoch)
            cur_epoch = epoch
            cur_loss, cur_acc, best_acc = test()
            pbar.set_description('[Epoch: {}| Loss: {:.4f}| Accuracy: {:.2f}| Best Accuracy: {:.2f}]'.format(cur_epoch, cur_loss, cur_acc, best_acc))
