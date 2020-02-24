import io
import pprint
import struct
import sys
import warnings

import cv2
import onnx

import ops

"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(g.input)-1: input nodes
    len(g.input)~ : other (hidden) nodes
"""

# XXX: Heuristics for scaling: only scale biases and the input
SCALE = 16
NUM_SLOTS = 2
INTERMEDIATE_VALUES_SIZE = 65536


def _Q15(num):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    if num < lower or num >= upper:
        if num != 1.0:
            warnings.warn(
                'Number {} goes beyond the range of _q15 ({}, {})'.format(
                    num, lower, upper))
        num = max(min(num, upper), lower)

    return int(num * 2 ** 15)


onnx_model = onnx.load(sys.argv[1])
g = onnx_model.graph
names = {}
n_input = len(g.input)
print("n_input = {}".format(n_input))

conv_param_names = set()

for idx, inp in enumerate(g.input):
    names[inp.name] = idx

for idx, n in enumerate(g.node):
    if n.op_type == 'Dropout':
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    assert len(output) == 1
    if n.op_type == 'Conv':
        for inp in n.input:
            conv_param_names.add(inp)
    names[output[0]] = idx + n_input

pprint.pprint(names)

model = []
for n in g.node:
    if n.op_type == 'MaxPool':
        stride = next(attr.ints[0] for attr in n.attribute if attr.name == 'strides')
        op_type = f'MaxPool_{stride}'
    else:
        op_type = n.op_type
    model.append(([names[i] for i in n.input], op_type))
parameters = [None for _ in range(n_input)]

for params in g.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params

def to_bytes(i, size=16):
    if size == 16:
        return struct.pack('h', i)
    elif size == 32:
        return struct.pack('i', i)
    elif size == 64:
        return struct.pack('q', i)
    else:
        raise ValueError(f'Unsupported size {size}')


def nchw2nhwc(arr, dims):
    N, C, H, W = dims
    ret = [0] * (N * C * H * W)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    old_idx = n * C * H * W + c * H * W + h * W + w
                    new_idx = n * H * W * C + h * W * C + w * C + c
                    ret[new_idx] = arr[old_idx]
    return ret, (N, H, W, C)


outputs = {
    'inputs': io.BytesIO(),
    'parameters': io.BytesIO(),
    'model': io.BytesIO(),
}

outputs['model'].write(to_bytes(len(model)))
outputs['model'].write(to_bytes(n_input))
parameters_bin_offset = 0
for inputs, op_type in model:
    outputs['model'].write(to_bytes(len(inputs)))
    outputs['model'].write(to_bytes(outputs['inputs'].tell()))  # Node.inputs_offset
    for inp in inputs:
        # the lowest bit is used as a flag in topological sort
        outputs['inputs'].write(to_bytes(inp * 2))
    outputs['model'].write(to_bytes(ops.ops[op_type]))
    outputs['model'].write(to_bytes(0))  # Node.scheduled

def bitwidth_and_flags_for_parameters(bitwidth):
    # Keep this in sync with common.h
    FLAG_SLOTS = 0b11
    FLAG_SLOTS_WIDTH = 2

    return bitwidth << FLAG_SLOTS_WIDTH | FLAG_SLOTS

for params in parameters:
    outputs['model'].write(to_bytes(parameters_bin_offset, size=32))  # params_offset
    if params is None:  # input
        im = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
        # See https://github.com/microsoft/CNTK/blob/master/Tutorials/CNTK_103*
        # for data format
        im = 255 - im
        im = im / 256  # to fit into range of _q15
        dimX, dimY = im.shape
        outputs['model'].write(to_bytes(dimX * dimY * 2, size=32))  # A _q15 is 16-bit
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                outputs['parameters'].write(to_bytes(_Q15(im[i, j] / SCALE)))
                parameters_bin_offset += 2
        outputs['model'].write(to_bytes(bitwidth_and_flags_for_parameters(16)))  # bitwidth_and_flags
        # extend_dims
        outputs['model'].write(to_bytes(1))
        outputs['model'].write(to_bytes(dimX))
        outputs['model'].write(to_bytes(dimY))
        outputs['model'].write(to_bytes(1))
    else:
        assert len(params.dims) <= 4
        reordered_dims = params.dims
        if params.data_type == onnx.TensorProto.FLOAT:
            if params.float_data:
                float_data = params.float_data
            else:
                float_data = [None] * (len(params.raw_data) // 4)
                for i in range(len(params.raw_data) // 4):
                    float_data[i] = struct.unpack_from(
                        'f', params.raw_data, offset=4 * i)[0]
            data_len = len(float_data)
            assert data_len > 0
            outputs['model'].write(to_bytes(data_len * 2, size=32))  # A _q15 is 16-bit
            if params.name in conv_param_names:
                print(f'Reorder conv param {params.name}')
                float_data_reordered, reordered_dims = nchw2nhwc(float_data, params.dims)
            else:
                float_data_reordered = float_data
            for param in float_data_reordered:
                if len(params.dims) != 4:  # most likely biases
                    outputs['parameters'].write(to_bytes(_Q15(param / SCALE)))
                else:
                    outputs['parameters'].write(to_bytes(_Q15(param)))
                parameters_bin_offset += 2
            outputs['model'].write(to_bytes(bitwidth_and_flags_for_parameters(16)))  # bitwidth_and_flags
        elif params.data_type == onnx.TensorProto.INT64:
            data_len = len(params.int64_data)
            outputs['model'].write(to_bytes(data_len * 8, size=32))
            for param in params.int64_data:
                outputs['parameters'].write(to_bytes(param, size=64))
                parameters_bin_offset += 8
            outputs['model'].write(to_bytes(bitwidth_and_flags_for_parameters(64)))  # bitwidth_and_flags
        else:
            assert False
        print('dims = {}, length = {}'.format(reordered_dims, data_len))
        for dim in reordered_dims:
            outputs['model'].write(to_bytes(dim))
        # dims are always 4 uint16_t's in C
        for _ in range(4 - len(reordered_dims)):
            outputs['model'].write(to_bytes(0))

# Placeholder for ParameterInfo of intermediate values
for idx, n in enumerate(g.node):
    outputs['model'].write(to_bytes(0, size=32))  # params_offset
    outputs['model'].write(to_bytes(0, size=32))  # params_len
    outputs['model'].write(to_bytes(0))  # bitwidth_and_flags
    for _ in range(4):  # dims[4]
        outputs['model'].write(to_bytes(0))

output_c = '''
#include "data.h"
'''
output_h = f'''
#pragma once
#include <stdint.h>

// const is for putting data on NVM
#ifdef __MSP430__
#  define GLOBAL_CONST const
#else
#  define GLOBAL_CONST
#endif

#define SCALE {SCALE}
#define NUM_SLOTS {NUM_SLOTS}
#define INTERMEDIATE_VALUES_SIZE {INTERMEDIATE_VALUES_SIZE}
'''
for var_name, data_obj in outputs.items():
    var_name += '_data'
    data_obj.seek(0)
    data = data_obj.read()
    output_h += f'''
extern GLOBAL_CONST uint8_t *{var_name};
#define {var_name.upper()}_LEN {len(data)};
'''
    output_c += f'''
#pragma NOINIT(_{var_name})
GLOBAL_CONST uint8_t _{var_name}[{len(data)}] = {{'''
    output_c += ', '.join([hex(b) for b in data])
    output_c += f'''}};
GLOBAL_CONST uint8_t *{var_name} = _{var_name};
'''

with open('data.c', 'w') as f, open('data.h', 'w') as g:
    f.write(output_c)
    g.write(output_h)

with open('nvm.bin', 'wb') as f:
    f.seek(256 * 1024 - 1)
    f.write(b'\0')
    f.seek(NUM_SLOTS * INTERMEDIATE_VALUES_SIZE)
    for data_obj in outputs.values():
        data_obj.seek(0)
        f.write(data_obj.read())