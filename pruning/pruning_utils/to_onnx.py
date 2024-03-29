from torch.autograd import Variable
#from torchsummary import summary
from torchinfo import summary
from thop import profile
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
            help='the network structure: KWS | KWS_CNN_S | HAR | mnist | LeNet_5 | SqueezeNet')
    parser.add_argument('--method', action='store', default='intermittent',
            help='pruning method: intermittent | energy')
    parser.add_argument('--model', action='store', default=None)
    parser.add_argument('--layout', action='store', default='nhwc', help='Select data layout: nhwc | nchw')
    parser.add_argument('--debug', action='store_true', help='Select data layout: nhwc | nchw')
    args = parser.parse_args()
    printArgs(args)

    if args.arch == 'HAR':
        input_shape = (9,1,128)
        model = models.HAR_CNN(None)
        dummy_input = Variable(torch.randn(1,9,1,128))
    elif args.arch == 'KWS_CNN_S':
        input_shape = (1,49,10)
        model = models.KWS_CNN_S(None)
        dummy_input = Variable(torch.randn(1,1,49,10))
    elif args.arch == 'SqueezeNet':
        input_shape = (3,32,32)
        model = models.SqueezeNet(None)
        dummy_input = Variable(torch.randn(1, 3, 32, 32))

    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    summary(model, input_shape, device='cpu')
    print(profile(model, inputs=(dummy_input, )))
    # save onnx model
    converted_name = "../onnx_models/{}/{}.onnx".format(args.method, args.arch)
    if args.arch == 'KWS_CNN_S':
        torch.onnx.export(model, dummy_input, converted_name, opset_version=OPSET, training=2) # exported in training mode to avoid fusing the conv and batchnormalization
    else:
        torch.onnx.export(model, dummy_input, converted_name, opset_version=OPSET)
    print('Converted model: {}'.format(converted_name))
    get_jobs(converted_name, args)
