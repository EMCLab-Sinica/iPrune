from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import argparse
import numpy as np
from torchsummary import summary

import models

def printArgs(args):
    print('\n => Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('======================\n')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch LeNet_5')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default=None)
    args = parser.parse_args()
    printArgs(args)

    if args.arch == 'LeNet_5':
        model = models.LeNet_5(None)

    dummy_input = Variable(torch.randn(1, 1, 28, 28))
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    state_dict = torch.load(args.pretrained)['state_dict']
    model.load_state_dict(state_dict)
    summary(model, (1, 28, 28), device='cpu')
    # save onnx model
    converted_name = "./onnx_models/{}.onnx".format(args.arch)
    torch.onnx.export(model, dummy_input, converted_name)
    print('Converted model: {}'.format(converted_name))
