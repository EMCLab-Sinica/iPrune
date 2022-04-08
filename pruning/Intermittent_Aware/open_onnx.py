import onnx, onnx.numpy_helper
import argparse
import numpy as np
import os
import sys
import logging
from scipy.sparse import bsr_matrix
from config import config

cwd = os.getcwd()
sys.path.append(cwd+'/../')

from util import *

logger = logging.getLogger(__name__)
def set_logger(args):
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    level = args.debug
    if level == 1:
        # debug
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif level == 0:
        # info
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)

    logger.addHandler(ch)
    return logger

def printArgs(args):
    print('\n => Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('======================\n')
    return

def im2col(tensor, shape):
    matrix = tensor.reshape(shape[0], -1)
    return matrix

def toBSR(matrix, group_size):
    bsr = bsr_matrix(matrix, blocksize=group_size)
    bsr.sort_indices()
    return bsr

def print_matrix(matrix):
    for row in matrix:
        logger.info(" ".join("{:.2f}".format(x) for x in row))
        logger.info("\n")

def getVal(node, colIdx):
    bsr = node['weights']
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr

def getJob(node, output_shape, group_size):
    bsr = node['weights']
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr
    if len(node['dims']) == 4:
        logger.debug('data: {}'.format(data))
        logger.info('cols: {}'.format(cols))
        logger.info('rows: {}'.format(rows))
        return len(cols) * output_shape[2] * output_shape[3] * group_size[0]
    elif len(node['dims']) == 2:
        logger.debug('data: {}'.format(data))
        logger.info('cols: {}'.format(cols))
        logger.info('rows: {}'.format(rows))
        return len(cols) * group_size[0]

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

def get_jobs(onnx_model, args):
    model = onnx.load(onnx_model)
    main_names = [n.input[1] for idx, n in enumerate(model.graph.node) if n.op_type == 'Conv' or n.op_type == 'Gemm']

    graph = []
    nodes = model.graph.initializer
    node_idx = 0
    for idx, node in enumerate(nodes):
        shape = node.dims
        logger.info(shape)
        matrix = onnx.numpy_helper.to_array(node)
        if node.name in main_names:
            layer_config = config[args.arch][node_idx]
            logger.debug(layer_config)
            if len(shape) == 4:
                if args.layout == 'nchw':
                    group_size = (layer_config['group'][0], layer_config['group'][1] * layer_config['filter'][2] * layer_config['filter'][3])
                    matrix = im2col(matrix, shape)
                elif args.layout == 'nhwc':
                    group_size = (layer_config['group'][0], layer_config['group'][1])
                    matrix = im2col(nchw2nhwc(matrix), shape)
            else:
                group_size = (layer_config['group'][0], layer_config['group'][1])
                matrix = im2col(matrix, shape)
            # print_matrix(matrix)
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
    node_idx = 0;
    total_job = 0
    for idx, n in enumerate(model.graph.node):
        if n.op_type == 'Conv' or n.op_type == 'Gemm':
            job = getJob(graph[node_idx], output_shapes[node_idx], config[args.arch][node_idx]['group'])
            total_job += job
            node_idx += 1
    print('total_job: {}'.format(total_job))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model', action='store', default=None)
    parser.add_argument('--arch', action='store', default='LeNet_5', help='the network architecture: LeNet_5 | SqueezeNet')
    parser.add_argument('--layout', action='store', default='nhwc', help='Select data layout: nhwc | nchw')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    set_logger(args)
    printArgs(args)
    get_jobs(args.onnx_model, args)

