import onnx, onnx.numpy_helper
import argparse
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix
from config import config

cwd = os.getcwd()
sys.path.append(cwd+'/../')

from util import *

def printArgs(args):
    print('\n => Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('======================\n')
    return

def lowering(tensor, shape):
    matrix = tensor.reshape(shape[0], -1)
    return matrix

def toBSR(matrix, group_size):
    '''
    append_size = width - matrix.shape[1] % width
    if append_size != width:
        matrix = np.concatenate((matrix, np.zeros((len(matrix), append_size))), 1)
    '''
    print(group_size)
    bsr = csr_matrix(matrix).tobsr(group_size)
    return bsr

def getVal(node, colIdx):
    bsr = node['weights']
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr

def getJob(node, output_shape):
    bsr = node['weights']
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr
    if len(node['dims']) == 4:
        print('cols: {}'.format(len(cols)))
        return len(cols) * output_shape[0] * output_shape[1]
    elif len(node['dims']) == 2:
        print('cols: {}'.format(len(cols)))
        return len(cols)

def printGroups(node):
    bsr = node['weights']
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr
    colIdx = 0
    for i in range(len(rows) - 1):
        cnt = rows[i + 1] - rows[i]
        print('----------------------------------------------------')
        print('filter {}:'.format(i))
        while(cnt):
            col = cols[colIdx]
            row = i
            print('group {}: {}'.format(col, data[colIdx]))
            cnt -= 1
            colIdx += 1

def nchw2nhwc(arr):
    arr = np.transpose(arr, axes=(0, 2, 3, 1))  # NCHW -> NHWC
    return arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model', action='store', default=None)
    parser.add_argument('--arch', action='store', default='LeNet_5', help='the network architecture: LeNet_5 | SqueezeNet')
    parser.add_argument('--group', action='store', type=int, default=[2, 1], help='Group size')
    parser.add_argument('--layout', action='store', default='nchw', help='Select data layout: nhwc | nchw')
    args = parser.parse_args()
    printArgs(args)

    graph = []

    model = onnx.load(args.onnx_model)
    '''
    for idx, n in enumerate(model.graph.input):
        print(n)
    for idx, n in enumerate(model.graph.node):
        print(n.op_type)
        print(n)
    '''
    main_names = [n.input[1] for idx, n in enumerate(model.graph.node) if n.op_type == 'Conv' or n.op_type == 'Gemm']

    nodes = model.graph.initializer
    node_idx = 0
    for idx, node in enumerate(nodes):
        shape = node.dims
        print(shape)
        matrix = onnx.numpy_helper.to_array(node)
        if node.name in main_names:
            layer_config = config[args.arch][node_idx]
            print(layer_config)
            matrix = lowering(matrix, shape)
            if len(shape) == 4:
                group_size = (layer_config['group'][0], layer_config['group'][1] * layer_config['filter'][0] * layer_config['filter'][1])
            else:
                group_size = (layer_config['group'][0], layer_config['group'][1])
            matrix = toBSR(matrix, group_size)
            sparse_node = {
                'dims': shape,
                'weights': matrix
            }
            graph.append(sparse_node)
            node_idx += 1

    # getVal(graph[], 0)
    # printGroups(graph[2])
    model_info = config[args.arch]
    output_shapes = [layer['output'] for layer in model_info]
    print(output_shapes)
    node_idx = 0;
    total_job = 0
    for idx, n in enumerate(model.graph.node):
        if n.op_type == 'Conv' or n.op_type == 'Gemm':
            job = getJob(graph[node_idx], output_shapes[node_idx])
            total_job += job
            node_idx += 1
    print('total_job: {}'.format(total_job))






