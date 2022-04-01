from torch.autograd import Variable
from torchsummary import summary
from open_onnx import *
import torch.onnx
import torchvision
import torch
import argparse
import numpy as np

import models

OPSET = 13

def printArgs(args):
    print('\n => Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('======================\n')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch LeNet_5')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the network structure: KWS | HAR | mnist | LeNet_5 | SqueezeNet')
    parser.add_argument('--model', action='store', default=None)
    parser.add_argument('--layout', action='store', default='nhwc', help='Select data layout: nhwc | nchw')
    parser.add_argument('--debug', action='store_true', help='Select data layout: nhwc | nchw')
    args = parser.parse_args()
    printArgs(args)

    if args.arch == 'LeNet_5':
        input_shape = (1,28,28)
        model = models.LeNet_5(None)
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
    elif args.arch == 'mnist':
        input_shape = (1,28,28)
        model = models.MNIST(None)
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
    elif args.arch == 'HAR':
        input_shape = (9,1,128)
        model = models.HAR_CNN(None)
        dummy_input = Variable(torch.randn(1,9,1,128))
    elif args.arch == 'KWS':
        input_shape = (1,25,10)
        model = models.KWS_DNN_S(None)
        dummy_input = Variable(torch.randn(1,25,10))
    elif args.arch == 'SqueezeNet':
        input_shape = (3,32,32)
        model = models.SqueezeNet(None)
        dummy_input = Variable(torch.randn(1, 3, 32, 32))

    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    summary(model, input_shape, device='cpu')
    # save onnx model
    converted_name = "./onnx_models/{}.onnx".format(args.arch)
    torch.onnx.export(model, dummy_input, converted_name, opset_version=OPSET)
    print('Converted model: {}'.format(converted_name))
    get_jobs(converted_name, args)
