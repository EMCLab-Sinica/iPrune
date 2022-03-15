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
            help='the network structure: LeNet_5 | SqueezeNet')
    parser.add_argument('--pretrained', action='store', default=None)
    args = parser.parse_args()
    printArgs(args)

    if args.arch == 'LeNet_5':
        input_shape = (1,28,28)
        model = models.LeNet_5(None)
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
    elif args.arch == 'LeNet_5_p':
        input_shape = (1,28,28)
        model = models.LeNet_5_p(None)
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
    elif args.arch == 'SqueezeNet':
        input_shape = (3,32,32)
        model = models.SqueezeNet(None)
        dummy_input = Variable(torch.randn(1, 3, 32, 32))

    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    state_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    summary(model, input_shape, device='cpu')
    # save onnx model
    if args.arch == 'LeNet_5_p':
        converted_name = "./onnx_models/{}.onnx".format('LeNet_5')
    else:
        converted_name = "./onnx_models/{}.onnx".format(args.arch)
    torch.onnx.export(model, dummy_input, converted_name)
    print('Converted model: {}'.format(converted_name))
