#!/usr/bin/python
import argparse
import ctypes
import dataclasses
import io
import itertools
import logging
import math
import os.path
import pathlib
import pprint
import struct
import textwrap
import warnings
import sys
import os
# warnings.simplefilter("ignore", UserWarning)
from typing import List
from scipy.sparse import bsr_matrix

import onnx
import onnx.defs
import onnx.helper
import onnx.numpy_helper
import numpy as np

cwd = os.getcwd()

from configs import configs
sys.path.append(cwd + '/../')
from pruning.config import config as model_configs
from utils import extract_data, find_initializer, find_node_by_output, find_node_by_input, find_tensor_value_info, load_model, get_model_ops, OPS_WITH_MERGE, DataLayout

logging.basicConfig()
logger = logging.getLogger(__name__)


"""
Goal: Mapping name-based nodes to integer-based ones.
Indexing policy:
    0~len(onnx_model.graph.input)-1: input nodes
    len(onnx_model.graph.input)~ : other (hidden) nodes
"""

class Constants:
    SLOT_PARAMETERS = 0xf0
    SLOT_TEST_SET = 0xff
    SLOT_CONSTANTS_MIN = SLOT_PARAMETERS
    SLOT_INTERMEDIATE_VALUES = 0b01
    NODE_NAME_LEN = 60
    EXTRA_INFO_LEN = 3  # for memory alignment
    TURNING_POINTS_LEN = 8
    MODEL_NODES_LEN = 0
    INPUTS_DATA_LEN = 0
    MAX_N_COL_FC = 0
    MAX_N_COL_CONV = 0
    MAX_ROW_LEN_FC = 0
    MAX_ROW_LEN_CONV = 0
    MAX_N_FILTER_GROUP = 0
    NUM_INPUTS = 0  # will be filled during parsing
    N_INPUT = 0
    # Match the size of external FRAM
    NVM_SIZE = 512 * 1024
    N_SAMPLES = 20
    # to make the code clearer; used in Conv
    TEMP_FILTER_WIDTH = 1
    LEA_BUFFER_SIZE = 0
    CPU_BUFFER_SIZE = 0
    ARM_PSTATE_LEN = 8704
    USE_ARM_CMSIS = 0
    CONFIG = None

    DEFAULT_TILE_H = 32
    BATCH_SIZE = 1
    STATEFUL = 0
    HAWAII = 0
    JAPARI = 0
    INTERMITTENT = 0
    INDIRECT_RECOVERY = 0
    METHOD = "Baseline"
    FIRST_SAMPLE_OUTPUTS = []
    # Sparse Matrix
    SPARSE = 0
    # exeuction model on stable power
    STABLE_POWER = 0
    # generate parameter.bin if true (--pbin)
    param_bin = 0
# XXX: Transpose does nothing as we happens to need NHWC
inplace_update_ops = ['Reshape', 'Softmax', 'Squeeze', 'Transpose', 'Unsqueeze']

audio_ops = ['DecodeWav', 'AudioSpectrogram', 'Mfcc']

other_flags = [
    # node flags
    'NHWC2NCHW',

    # parameter flags
    'CHANNEL_FIRST',
    'SEPARATE_TILING',  # Tiles in different channels are actually in different slots
]

def op_flag(flag):
    return 2 ** other_flags.index(flag)

def _Q15(arr, name):
    """Transform a floating point number to TI's fixed point _q15 format"""

    # See DSPLib_1_30_00_02/include/DSPLib_support.h

    lower = -1
    upper = 32767.0 / 32768.0

    overflowed_indices = np.concatenate((
        np.flatnonzero(np.asarray(arr < lower)),
        np.flatnonzero(np.asarray(arr > upper))
    ))
    for idx in overflowed_indices:
        warnings.warn(f'{name} value {arr[idx]} goes beyond the range of _q15 ({lower}, {upper})')

    arr = np.minimum(np.maximum(arr, lower), upper)

    return (arr * 2 ** 15).astype(int)

# https://stackoverflow.com/a/11481471/3786245
class ConvNodeFlags(ctypes.Structure):
    _fields_ = [
        ("input_tile_c", ctypes.c_uint16),
        ("output_tile_c", ctypes.c_uint16),
        ("output_tile_w", ctypes.c_uint16),
        ("output_tile_h", ctypes.c_uint16),
        ("pads", ctypes.c_uint8 * 4),
    ]

class MaxPoolFlags(ctypes.Structure):
    _fields_ = [
        ("kernel_shape", ctypes.c_uint8 * 2),
        ("strides", ctypes.c_uint8 * 2),
    ]

class GemmNodeFlags(ctypes.Structure):
    _fields_ = [
        ("tile_channel", ctypes.c_uint16, 16),
    ]

class GemmMergeNodeFlags(ctypes.Structure):
    _fields_ = [
        ("tile_length", ctypes.c_uint16, 16),
    ]

class SqueezeNodeFlags(ctypes.Structure):
    _fields_ = [
        ("axes", ctypes.c_uint8, 8),  # a bitmap for axes to squeeze/unsqueeze
    ]

class ExtraNodeFlags(ctypes.Union):
    _fields_ = [
        ("conv", ConvNodeFlags),
        ("maxpool", MaxPoolFlags),
        ("gemm", GemmNodeFlags),
        ("gemmmerge", GemmMergeNodeFlags),
        ("squeeze", SqueezeNodeFlags),
    ]

class NodeFlags_bits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("generic", ctypes.c_uint8, 8),
        ("kernel_size", ctypes.c_uint8, 8),
        ("stride", ctypes.c_uint8 * 2), # stride_H/stride_W
        ("extra", ExtraNodeFlags),
    ]

class NodeFlags(ctypes.Union):
    _fields_ = [
        ("b", NodeFlags_bits),
        ("as_bytes", ctypes.c_uint8 * 16),
    ]

    def __repr__(self):
        ret = '<NodeFlags'
        for field in NodeFlags_bits._fields_:
            key = field[0]
            ret += f' {key}={getattr(self.b, key)}'
        ret += '>'
        return ret

class ONNXNodeWrapper:
    def __init__(self, orig_node: onnx.NodeProto):
        self.orig_node = orig_node
        self.flags = NodeFlags()

    def __getattr__(self, name):
        return getattr(self.orig_node, name)


def get_prev_node(n):
    return nodes[names[n.input[0]] - Constants.N_INPUT]

lea_buffer_size = {
    # (4096 - 0x138 (LEASTACK) - 2 * 8 (MSP_LEA_MAC_PARAMS)) / sizeof(int16_t)
    'msp430': 1884,
    # determined by trial and error
    'msp432': 18000,
}

cpu_buffer_size = {
    # determined by trial and error
    'msp430': 800,
    'msp432': 18000,
}

parser = argparse.ArgumentParser()
parser.add_argument('config', choices=configs.keys())
parser.add_argument('--all-samples', action='store_true')
parser.add_argument('--write-images', action='store_true')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--target', choices=('msp430', 'msp432'), required=True)
parser.add_argument('--method', default='intermittent',
                    help='choose pruned models: energy | intermittent')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--sparse', action='store_true')
parser.add_argument('--stable-power', action='store_true')
parser.add_argument('--pbin', action='store_true')
intermittent_methodology = parser.add_mutually_exclusive_group(required=True)
intermittent_methodology.add_argument('--baseline', action='store_true')
intermittent_methodology.add_argument('--hawaii', action='store_true')
intermittent_methodology.add_argument('--japari', action='store_true')
intermittent_methodology.add_argument('--stateful', action='store_true')
args = parser.parse_args()
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)
config = configs[args.config]
config['total_sample_size'] = np.prod(config['sample_size'])
if 'gemm_tile_length' not in config:
    config['gemm_tile_length'] = 0
Constants.CONFIG = args.config
Constants.FIRST_SAMPLE_OUTPUTS = config['first_sample_outputs']
if args.all_samples:
    Constants.N_SAMPLES = config['n_all_samples']
    Constants.NVM_SIZE += config['n_all_samples'] * 2*config['total_sample_size']  # multiply by 2 for Q15
model_data = config['data_loader'](start=0, limit=Constants.N_SAMPLES)

if args.stateful:
    Constants.STATEFUL = 1
    Constants.METHOD = "STATEFUL"
if args.hawaii:
    Constants.HAWAII = 1
    Constants.METHOD = "HAWAII"
if args.japari:
    Constants.JAPARI = 1
    Constants.METHOD = "JAPARI"
    config['intermediate_values_size'] *= 2
if args.sparse:
    Constants.SPARSE = 1
if args.stable_power:
    Constants.STABLE_POWER = 1
Constants.INTERMITTENT = Constants.STATEFUL | Constants.HAWAII | Constants.JAPARI
Constants.INDIRECT_RECOVERY = Constants.STATEFUL | Constants.JAPARI
if args.target == 'msp432':
    Constants.USE_ARM_CMSIS = 1
Constants.LEA_BUFFER_SIZE = lea_buffer_size[args.target]
Constants.CPU_BUFFER_SIZE = cpu_buffer_size[args.target]

if args.pbin:
    Constants.param_bin = 1

if args.config == 'pruned_cifar10':
    model_config = model_configs['SqueezeNet']
    Constants.CPU_BUFFER_SIZE = 400
elif args.config == 'pruned_har':
    model_config = model_configs['HAR']
elif args.config == 'pruned_kws_cnn':
    Constants.CPU_BUFFER_SIZE = 700
    model_config = model_configs['KWS_CNN_S']

onnx_model = load_model(config, args.method)
# print(onnx_model)
names = {}

def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None

# Remove Squeeze and Reshape nodes with constants as the input
replaced_nodes_map = {}

def replace_squeeze(node, inp):
    # Since opset 13, axes is an input instead of an attribute
    try:
        axes_name = node.input[1]
        axes = find_initializer(onnx_model, axes_name).int64_data
    except IndexError:
        axes = get_attr(node, 'axes')
    new_dims = [dim for dim_idx, dim in enumerate(inp.dims) if dim_idx not in axes]
    # Repeated fields cannot be assigned directly
    # https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
    inp.dims[:] = new_dims

def replace_reshape(node, inp):
    dims_name = node.input[1]
    new_dims = find_initializer(onnx_model, dims_name).int64_data
    assert new_dims
    inp.dims[:] = new_dims

replace_handlers = {
    'Squeeze': replace_squeeze,
    'Reshape': replace_reshape,
}

def replace_nodes():
    for n in onnx_model.graph.node:
        if n.op_type not in ('Squeeze', 'Reshape'):
            continue
        inp = find_initializer(onnx_model, n.input[0])
        if inp:
            replace_handlers[n.op_type](n, inp)
            replaced_nodes_map[n.output[0]] = n.input[0]
'''
Transpose matrix B:
dims: [n_filter, n_channel] -> [n_channel, n_filter]
'''
def transpose_gemm(onnx_model: onnx.ModelProto):
    for node in onnx_model.graph.node:
        if node.op_type != 'Gemm':
            continue
        transB = get_attr(node, 'transB')
        B = find_initializer(onnx_model, node.input[1])
        if transB != 1 or B is None:
            continue
        data = extract_data(B)
        data = np.transpose(data)
        B.CopyFrom(onnx.helper.make_tensor(B.name, B.data_type, (B.dims[1], B.dims[0]), np.concatenate(data)))
        for idx, attr in enumerate(node.attribute):
            if attr.name == 'transB':
                del node.attribute[idx]
                break

replace_nodes()
transpose_gemm(onnx_model)

main_names = [n.input[1] for idx, n in enumerate(onnx_model.graph.node) if n.op_type == 'Conv' or n.op_type == 'Gemm']

# Split Conv/Gemm into Conv/Gemm and ConvMerge/GemmMerge (for OFM scaling up and merge of OFMs from channel tiling)
new_nodes = []
for idx, n in enumerate(onnx_model.graph.node):
    if n.op_type in audio_ops:
        logger.warning('skipping audio operator %s', n.op_type)
        continue
    new_nodes.append(n)
    if n.op_type in OPS_WITH_MERGE:
        output_name = n.output[0]
        new_node = onnx.NodeProto()
        new_node.name = (n.name or n.op_type) + ':merge'
        new_node.op_type = n.op_type + 'Merge'
        new_node.input[:] = n.output[:] = [output_name + '_before_merge']
        new_node.output[:] = [output_name]
        new_nodes.append(new_node)

new_nodes = [n for n in new_nodes if n.output[0] not in replaced_nodes_map.keys()]
for n in new_nodes:
    for idx, inp in enumerate(n.input):
        n.input[idx] = replaced_nodes_map.get(inp, inp)

nodes = [ONNXNodeWrapper(n) for n in new_nodes]

conv_param_names = set()
gemm_param_names = set()

for idx, inp in enumerate(onnx_model.graph.input):
    names[inp.name] = idx

# For some ONNX models (e.g., squeezenet-cifar10 converted from Keras), inputs
# do not include initializers. Merge them here.
inputs_len = len(names.keys())
for idx, initializer in enumerate(onnx_model.graph.initializer):
    if initializer.name not in names:
        names[initializer.name] = idx + inputs_len

Constants.N_INPUT = len(names.keys())
logger.info('Constants.N_INPUT = %d', Constants.N_INPUT)

def infer_auto_pad(node):
    # https://github.com/onnx/onnx/blob/master/docs/Operators.md#conv
    conv_flags = node.flags.b.extra.conv
    auto_pad = get_attr(node, 'auto_pad')
    pads = get_attr(node ,'pads')
    if pads:
        assert len(pads) <= 4
        if args.config == 'pruned_har':
            # Since the conv1d is not supported for pytorch to onnx, I replace conv1d with conv2d.
            # However, the pad_H_left/pad_W_left and pad_H_right/pad_W_right_left should be the same in conv2d in pytorch.
            # Therefore, modify pad_W_right to 0 manually to avoid mistake.
            pads[3] = 0
        # https://stackoverflow.com/questions/4145775/how-do-i-convert-a-python-list-into-a-c-array-by-using-ctypes
        conv_flags.pads = (ctypes.c_uint8 * 4)(*pads)
    if auto_pad in (b'SAME_UPPER', b'SAME_LOWER'):
        kernel_shape = get_attr(node, 'kernel_shape')
        conv_flags.pads[0] = conv_flags.pads[2] = kernel_shape[0] // 2
        conv_flags.pads[1] = conv_flags.pads[3] = kernel_shape[1] // 2
        if conv_flags.pads[0]*2+1 != kernel_shape[0] or conv_flags.pads[1]*2+1 != kernel_shape[1]:
            raise NotImplementedError

for idx, n in enumerate(nodes):
    if n.op_type in ('Dropout', 'BatchNormalization'):
        output = n.output[:1]  # we don't care the second output `mask`
    else:
        output = n.output
    if n.op_type == 'Conv':
        conv_param_names.add(n.input[1])
        infer_auto_pad(n)
        strides = get_attr(n, 'strides')
        n.flags.b.stride = (ctypes.c_uint8*2)(*strides)
    if n.op_type == 'Gemm':
        gemm_param_names.add(n.input[1])
    if n.op_type == 'MaxPool':
        kernel_shape = get_attr(n, 'kernel_shape')  # this field is required
        assert len(kernel_shape) == 2
        n.flags.b.extra.maxpool.kernel_shape = (ctypes.c_uint8*2)(*kernel_shape)
        strides = get_attr(n, 'strides')
        if args.config == 'pruned_har':
            # Since the maxpool1d is not supported for pytorch to onnx, I replace maxpool1d with maxpool2d.
            # However, the stride_H and stride_W should be the same in maxpool2d in pytorch.
            # Therefore, modify stride_H to 1 manually to avoid mistake.
            strides[0] = 1
        if strides is not None:
            n.flags.b.extra.maxpool.strides = (ctypes.c_uint8*2)(*strides)
            n.flags.b.stride = (ctypes.c_uint8*2)(*strides)
        else:
            # "If not present, the stride defaults to 1 along each spatial axis."
            # https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
            n.flags.b.extra.maxpool.strides = (ctypes.c_uint8*2)(1, 1)
            n.flags.b.stride = (ctypes.c_uint8*2)(1, 1)
    if n.op_type == 'Reshape':
        prev_node = n
        while prev_node and prev_node.op_type in inplace_update_ops:
            prev_node = find_node_by_output(nodes, prev_node.input[0])
        if prev_node and prev_node.op_type in ('MaxPool', 'Relu'):
            prev_node.flags.b.generic += op_flag('NHWC2NCHW')
    if n.op_type in ('Squeeze', 'Unsqueeze'):
        axes = get_attr(n, 'axes') or []
        node_flags = n.flags.b.extra.squeeze
        node_flags.axes = 0
        for axis in axes:
            node_flags.axes |= (1 << axis)
    if n.op_type == 'GemmMerge':
        n.flags.b.extra.gemmmerge.tile_length = config['gemm_tile_length']
    for output_ in output:
        names[output_] = idx + Constants.N_INPUT

pprint.pprint(names)

@dataclasses.dataclass
class Node:
    name: str
    output_name: str
    inputs: List[int]
    op_type: str
    flags: NodeFlags
    max_output_id: int

def extend_for_footprints(n):
    return n + n // Constants.BATCH_SIZE

def determine_conv_tile_c(n, node_idx):
    logger.debug('Determine tile size for Conv node %s', n.name)

    output_value_info = find_tensor_value_info(onnx_model, n.output[0])
    filter_info = find_initializer(onnx_model, n.input[1])
    node_flags = n.flags.b.extra.conv

    if model_config:
        is_separate_tiling = False
        if not find_initializer(onnx_model, n.input[0]):
            input_node = find_node_by_output(onnx_model.graph.node, n.input[0])
            if input_node and input_node.op_type == 'Concat':
                is_separate_tiling = True

        shape = output_value_info.type.tensor_type.shape
        OUTPUT_CHANNEL = shape.dim[1].dim_value
        OUTPUT_H = shape.dim[2].dim_value
        OUTPUT_W = shape.dim[3].dim_value
        CHANNEL = filter_info.dims[1]
        kH = filter_info.dims[2]
        kW = filter_info.dims[3]
        tile_kH = model_config[node_idx]['tile']['weight'][2]
        tile_kW = model_config[node_idx]['tile']['weight'][3]

        def output2input(tile_size, kernel_size, stride):
            return (tile_size - 1) * stride + kernel_size

        output_tile_h = model_config[node_idx]['tile']['output'][2]
        input_tile_h = output2input(tile_size=model_config[node_idx]['tile']['output'][2], \
                                    kernel_size=model_config[node_idx]['filter'][2],
                                    stride=model_config[node_idx]['stride'][0])
        output_tile_w = model_config[node_idx]['tile']['output'][3]
        input_tile_w = output2input(tile_size=model_config[node_idx]['tile']['output'][3], \
                                    kernel_size=model_config[node_idx]['filter'][3],
                                    stride=model_config[node_idx]['stride'][1])
        output_tile_c = model_config[node_idx]['tile']['output'][1]

        max_continuous_channels = CHANNEL
        if is_separate_tiling:
            max_continuous_channels //= 2
        node_flags.input_tile_c = model_config[node_idx]['tile']['input'][1]

        logger.debug('Initial input_tile_c=%d', node_flags.input_tile_c)
        # ignore the code if you want to fix tile size
        def get_memory_usage(output_tile_h, output_tile_w, input_tile_h, input_tile_w, output_tile_c, input_tile_c, filter_len):
            # *2 as in JAPARI, the number of footprint weights is up to the number of
            # filters (e.g., batch size=1)
            weight_memory_usage = ((output_tile_c + 1) + Constants.TEMP_FILTER_WIDTH) * filter_len
            input_memory_usage = (input_tile_h + 1) * input_tile_w * input_tile_c
            output_memory_usage = output_tile_h * output_tile_w * output_tile_c
            logger.debug('Checking output_tile_h=%d, output_tile_w=%d, input_tile_h=%d, input_tile_w=%d', \
                         output_tile_h, output_tile_w, input_tile_h, input_tile_w)
            logger.debug('Checking output_tile_c=%d, input_tile_c=%d, filter_len=%d', output_tile_c, input_tile_c, filter_len)
            logger.debug('Checking memory usage: weight=%d, input=%d, output=%d, total=%d', \
                         weight_memory_usage, input_memory_usage, output_memory_usage, weight_memory_usage+input_memory_usage+output_memory_usage)
            assert(output_memory_usage < Constants.CPU_BUFFER_SIZE)
            return weight_memory_usage + input_memory_usage + output_memory_usage

        # inner +1 for biases
        filter_len = ((node_flags.input_tile_c * tile_kW + 1) + 1) // 2 * 2 * tile_kH
        input_tile_c =((node_flags.input_tile_c + 1) + 1) // 2 * 2
        while get_memory_usage(output_tile_h, output_tile_w, \
                               input_tile_h, input_tile_w, \
                               output_tile_c, input_tile_c, filter_len) > Constants.LEA_BUFFER_SIZE:
            logger.debug('output_tile_w=%d, input_tile_w=%d', output_tile_w, input_tile_w)
            output_tile_w -= 1
            input_tile_w -= 1
            if output_tile_w < 1 or input_tile_w < 1:
                print("Input channel or output channel may be too large")
                exit()
        node_flags.output_tile_w = output_tile_w
        node_flags.output_tile_h = output_tile_h
        node_flags.output_tile_c = output_tile_c
    else:
        print("Please select configed model.")
        exit()

    '''
    print('input_tile_c: {}'.format(node_flags.input_tile_c))
    print('output_tile_c: {}'.format(node_flags.output_tile_c))
    print('output_tile_w: {}'.format(node_flags.output_tile_w))
    print('output_tile_h: {}'.format(node_flags.output_tile_h))
    '''

def determine_gemm_tile_sizes(n, node_idx):
    logger.debug('Determine tile size for Gemm node %s', n.name)

    A = find_tensor_value_info(onnx_model, n.input[0])
    B = find_initializer(onnx_model, n.input[1])
    A_shape = A.type.tensor_type.shape
    A_rows = 1  # Not using A_shape.dim[0] here, as it's a symbol "N"
    A_cols = A_shape.dim[1].dim_value
    B_rows = B.dims[0]
    node_flags = n.flags.b.extra.gemm

    # writing a batch at a time is simpler and faster
    tile_size_unit = config['op_filters']
    if model_config == None:
        while True:
            # LEA wants addresses to be 4 byte-aligned, or 2 Q15-aligned
            node_flags.tile_channel = min([(Constants.ARM_PSTATE_LEN / tile_size_unit) / 2 * 2 - 2, B_rows,
                                           (config['gemm_tile_length'] or float('inf'))]) // tile_size_unit * tile_size_unit
            full_tile_width = (extend_for_footprints(tile_size_unit)+1)/2*2
            while node_flags.tile_channel > 0:
                tmp = int(math.ceil(B_rows / node_flags.tile_channel))
                needed_mem = (A_rows * A_cols + 2) + (node_flags.tile_channel + 2) * full_tile_width + A_rows * full_tile_width
                logger.debug("tile_channel=%d, tmp=%d, needed_mem=%d", node_flags.tile_channel, tmp, needed_mem)
                if needed_mem <= Constants.LEA_BUFFER_SIZE:
                    break
                node_flags.tile_channel -= tile_size_unit
            logger.debug("tile_channel = %d", node_flags.tile_channel)
            if node_flags.tile_channel > 0:
                break
    else:
        # manually set tile size
        node_flags.tile_channel = model_config[node_idx]['group'][1]
    '''
    print('tile_channel: {}'.format(node_flags.tile_channel))
    print('tile_size_unit: {}'.format(tile_size_unit))
    '''

    assert tile_size_unit * (node_flags.tile_channel + 2) <= Constants.ARM_PSTATE_LEN

graph = []
node_idx = 0
for n in nodes:
    if n.op_type == 'Conv':
        determine_conv_tile_c(n, node_idx)
        node_idx += 1
    if n.op_type == 'Gemm':
        determine_gemm_tile_sizes(n, node_idx)
        node_idx += 1
    graph.append(Node(name=n.name or n.op_type,
                      output_name=n.output[0],
                      inputs=[names[i] for i in n.input],
                      op_type=n.op_type,
                      flags=n.flags,
                      max_output_id=0))

for idx, node in enumerate(graph):
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = graph[inp - Constants.N_INPUT]
        used_node.max_output_id = max([idx, used_node.max_output_id])

# Inputs of Concat should be kept until Concat is processed
for idx, node in enumerate(graph):
    if node.op_type != 'Concat':
        continue
    for inp in node.inputs:
        if inp < Constants.N_INPUT:
            continue
        used_node = graph[inp - Constants.N_INPUT]
        used_node.max_output_id = max([used_node.max_output_id, node.max_output_id])

parameters = [None for _ in range(Constants.N_INPUT)]

for params in onnx_model.graph.initializer:
    if params.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64):
        raise Exception('unsupported data type {}'.format(params.data_type))

    assert parameters[names[params.name]] is None
    parameters[names[params.name]] = params

pprint.pprint(graph)

def to_bytes(arr, size=16):
    if not np.shape(arr):
        arr = [arr]
    FORMAT_CHARS = {
        8: 'B',  # unsigned char
        16: 'h',
        32: 'i',
        64: 'q'
    }
    if size not in FORMAT_CHARS:
        raise ValueError(f'Unsupported size {size}')
    return struct.pack('%u%c' % (len(arr), FORMAT_CHARS[size]), *arr)

def nchw2nhwc(arr, dims):
    arr = np.reshape(arr, dims)  # Change flattened to 4-D
    arr = np.transpose(arr, axes=(0, 2, 3, 1))  # NCHW -> NHWC
    return arr.flatten()  # Change it back to flattened

def nchw2nhwc_without_flatten(arr):
    arr = np.transpose(arr, axes=(0, 2, 3, 1))  # NCHW -> NHWC
    return arr

def nchw2nwhc_without_flatten(arr):
    arr = np.transpose(arr, axes=(0, 3, 2, 1))  # NCHW -> NWHC
    return arr

def im2col(arr, dims):
    arr = np.reshape(arr, (dims[0], -1))
    return arr

if args.sparse:
    outputs = {
        'parameters': io.BytesIO(),
        'samples': io.BytesIO(),
        'model': io.BytesIO(),
        'nodes': io.BytesIO(),
        'model_parameters_info': io.BytesIO(),
        'intermediate_parameters_info': io.BytesIO(),
        'labels': io.BytesIO(),
        # for sparse model
        'rows': io.BytesIO(),
        'cols': io.BytesIO(),
        'first_tile_index': io.BytesIO()
    }
else:
    outputs = {
        'parameters': io.BytesIO(),
        'samples': io.BytesIO(),
        'model': io.BytesIO(),
        'nodes': io.BytesIO(),
        'model_parameters_info': io.BytesIO(),
        'intermediate_parameters_info': io.BytesIO(),
        'labels': io.BytesIO(),
    }

if Constants.param_bin:
    EXTERNAL_DATA = ('parameters',)

Constants.MODEL_NODES_LEN = len(graph)

model = outputs['model']
model.write(to_bytes(0))  # Model.running
model.write(to_bytes(0))  # Model.run_counter
model.write(to_bytes(0))  # Model.layer_idx
for _ in range(config['num_slots']): # Model.slots_info
    if Constants.INDIRECT_RECOVERY:
        model.write(to_bytes(1, size=8)) # SlotInfo.state_bit
        model.write(to_bytes(0, size=8)) # SlotInfo.n_turning_points
        for __ in range(Constants.TURNING_POINTS_LEN):
            model.write(to_bytes(-1))   # SlotInfo.turning_points
    model.write(to_bytes(-1))       # SlotInfo.user
model.write(to_bytes(0, size=8))  # Model.dummy
model.write(to_bytes(0, size=8))  # Model.version

if args.sparse:
    @dataclasses.dataclass
    class ParametersSlot:
        offset: int
        target: io.BytesIO
        cols_offset: int
        cols: io.BytesIO
        rows_offset: int
        rows: io.BytesIO
        first_tile_index_offset: int
        first_tile_index: io.BytesIO
        slot_id: int
else:
    @dataclasses.dataclass
    class ParametersSlot:
        offset: int
        target: io.BytesIO
        slot_id: int

if args.sparse:
    parameters_slot = ParametersSlot(offset=0, \
                                     target=outputs['parameters'], \
                                     cols_offset=0, \
                                     cols=outputs['cols'], \
                                     rows_offset=0, \
                                     rows=outputs['rows'], \
                                     first_tile_index_offset=0, \
                                     first_tile_index=outputs['first_tile_index'], \
                                     slot_id=Constants.SLOT_PARAMETERS)
else:
    parameters_slot = ParametersSlot(offset=0, target=outputs['parameters'], slot_id=Constants.SLOT_PARAMETERS)

output_nodes = outputs['nodes']
for node in graph:
    Constants.NUM_INPUTS = max(Constants.NUM_INPUTS, len(node.inputs))
logger.info('Maximum number of inputs = %d', Constants.NUM_INPUTS)

ops = get_model_ops(onnx_model)

def write_str(buffer: io.BytesIO, data: str):
    assert Constants.NODE_NAME_LEN >= len(data), f'String too long: {data}'
    buffer.write(data.encode('ascii') + b'\0' * (Constants.NODE_NAME_LEN - len(data)))

for node in graph:
    write_str(output_nodes, node.name)
    write_str(output_nodes, node.output_name)
    output_nodes.write(to_bytes(len(node.inputs)))
    for inp in node.inputs:
        output_nodes.write(to_bytes(inp))
    for _ in range(Constants.NUM_INPUTS - len(node.inputs)):
        output_nodes.write(to_bytes(0))
    output_nodes.write(to_bytes(node.max_output_id))
    output_nodes.write(to_bytes(ops.index(node.op_type)))
    assert ctypes.sizeof(node.flags.as_bytes) == ctypes.sizeof(node.flags.b), f'Node flags require {ctypes.sizeof(node.flags.b)} bytes'
    for idx in range(ctypes.sizeof(node.flags.as_bytes)):
        output_nodes.write(to_bytes(node.flags.as_bytes[idx], size=8))
    if Constants.HAWAII:
        for _ in range(2):
            output_nodes.write(to_bytes(0, size=64))  # Node::Footprint

parameter_info_idx = 0

def decode_raw_data(params):
    format_char = {
        onnx.TensorProto.FLOAT: 'f',
        onnx.TensorProto.INT64: 'q',
    }[params.data_type]
    return list(map(lambda t: t[0], struct.iter_unpack(format_char, params.raw_data)))

def dump_matrix(arr):
    logger.debug(arr.shape)
    for row in arr:
        logger.debug(" ".join("{:>6d}".format(x) for x in row))

def dump_matrix_list(arr):
    print(" ".join("{:>6}".format(x) for x in arr))

def xxxx2xcxxx(arr, config, dims):
    chunk_len = dims[1] # c
    arr = im2col(arr, dims)
    new_arr = []

    for row in arr:
        # avoid omiting 0 when transform the matrix to csr
        row = row + 1
        lists = [np.array(row[i : i + chunk_len]) for i in range(0, len(row), chunk_len)]
        lists = np.array(lists)
        group_size = (dims[2] * dims[3], config['group'][1])
        bsr = bsr_matrix(lists, blocksize=group_size)
        bsr.sort_indices()
        new_row = []
        for data in bsr.data:
            data = data.flatten() - 1
            new_row.extend(data)
        new_arr.append(new_row)
    return np.array(new_arr)

def toBSR(matrix, config, dims, op_type):
    shape = matrix.shape
    matrix = np.reshape(matrix, tuple(dims))
    if op_type == 'CONV':
        matrix = nchw2nwhc_without_flatten(matrix)
        matrix = xxxx2xcxxx(matrix, config, dims)
        # len(rows): #filter groups
        # len(cols): the number of input_tile_c * K * K
        group_size = (config['group'][0], config['group'][1])
        dump_matrix(matrix)
        bsr = bsr_matrix(matrix, blocksize=group_size)
        bsr.sort_indices()
        logger.debug('Data(transposed):\n{}'.format(bsr.data))
        data = bsr.data
    elif op_type == 'GEMM':
        # the dim of the GEMM matrix has been [n_channel, n_filter]
        # len(rows): the number of input_tile_c
        # len(cols): the number of filter groups
        group_size = (config['group'][1], config['group'][0])
        bsr = bsr_matrix(matrix, blocksize=group_size)
        bsr.sort_indices()
        data = bsr.data
    data = np.reshape(data, -1)
    cols = bsr.indices
    rows = bsr.indptr
    logger.debug('Data:\n{}'.format(data))
    logger.debug('Cols: {}'.format(cols))
    logger.debug('Rows: {}'.format(rows))
    logger.info('filter size: {}'.format(len(data)))
    logger.info('Rows size: {}'.format(rows.shape))
    logger.info('Cols size: {}'.format(cols.shape))
    if op_type == 'CONV':
        for i in range(1, len(rows)):
            n_col = rows[i] - rows[i - 1]
            Constants.MAX_N_COL_CONV = max(Constants.MAX_N_COL_CONV, n_col + 1)
        Constants.MAX_ROW_LEN_CONV = max(Constants.MAX_ROW_LEN_CONV, len(rows) + 1)
    if op_type == 'GEMM':
        for i in range(1, len(rows)):
            n_col = rows[i] - rows[i - 1]
            Constants.MAX_N_COL_FC = max(Constants.MAX_N_COL_FC, n_col + 1)
        Constants.MAX_ROW_LEN_FC = max(Constants.MAX_ROW_LEN_FC, len(rows) + 1)
        Constants.MAX_N_FILTER_GROUP = max(Constants.MAX_N_FILTER_GROUP, math.ceil(dims[1] / config['group'][0])  + 1)
    return data, cols, rows

def find_first_tile_index(cols, rows, config, dims, op_type):
    if op_type == 'GEMM':
        slice_n_output_tile_c = int(dims[1] / config['group'][0])
        row_len = len(rows)
        first_tile_index = [-1] * slice_n_output_tile_c
        for i in range(1, row_len):
            n_cols = rows[i] - rows[i - 1]
            cur_n_cols = 0
            while cur_n_cols < n_cols:
                col_index = rows[i - 1] + cur_n_cols
                col_val = cols[col_index]
                if first_tile_index[col_val] == -1:
                    first_tile_index[col_val] = i - 1
                cur_n_cols += 1
    logger.debug('first_tile_index: {}'.format(first_tile_index))
    logger.info('first_tile_index length: {}'.format(len(first_tile_index)))
    return first_tile_index

def get_float_data(param):
    if param.float_data:
        float_data = param.float_data
    else:
        float_data = decode_raw_data(param)
    return float_data

param_limits = {}
def get_param_limit(model, node, float_data):
    return max([abs(data) for data in float_data]) * 1.5

def write_scale(dest, scale):
    shift = 0
    while scale >= 1:
        shift += 1
        scale /= 2
    dest.write(to_bytes(int(scale*2**15)))             # scale.fract
    dest.write(to_bytes(shift, size=8))     # scale.shift
    dest.write(to_bytes(0, size=8))         # scale.dummy

model_parameters_info = outputs['model_parameters_info']
for params in parameters:
    if params is None:  # input
        # Actual data for test samples are added last
        dims = model_data.images[0].shape
        model_parameters_info.write(to_bytes(parameters_slot.offset, size=32))  # params_offset
        if Constants.param_bin:
            model_parameters_info.write(to_bytes(0, size=32))  # params_fram_offset
        model_parameters_info.write(to_bytes(np.prod(dims) * 2, size=32))  # A _q15 is 16-bit
        if args.sparse:
            model_parameters_info.write(to_bytes(0, size=32))  # cols_offset, the place is used by sparse matrix
            model_parameters_info.write(to_bytes(0, size=32))  # rows_offset, the place is used by sparse matrix
            model_parameters_info.write(to_bytes(0, size=32))  # first_tile_index_offset, the place is used by sparse matrix
        model_parameters_info.write(to_bytes(16, size=8))                # bitwidth
        model_parameters_info.write(to_bytes(Constants.SLOT_TEST_SET, size=8))     # slot
        # extend_dims
        model_parameters_info.write(to_bytes(1))
        if args.config == 'pruned_har':
            # expand the dims of input in order to fit the shape of conv2d.
            dims = np.insert(dims, 1, 1)
        elif args.config == 'pruned_cifar10':
            # Pruned cifar10 has no a transpose layer to transpose the data_layout from NCHW to NCHW.
            # Thus, assign directly the NHWC layout for pruned cifar10
            dims = (3, 32, 32)
        for dim in dims:
            model_parameters_info.write(to_bytes(dim))
        for _ in range(3 - len(dims)):
            model_parameters_info.write(to_bytes(0))
        write_scale(model_parameters_info, config['input_scale'])
    else:
        params_scale = 0
        assert len(params.dims) <= 4
        if params.data_type == onnx.TensorProto.FLOAT:
            float_data = get_float_data(params)
            if params.name in conv_param_names and not args.sparse:
                logger.info('Reorder conv param %s', params.name)
                float_data = nchw2nhwc(float_data, params.dims)

            used_node = find_node_by_input(onnx_model.graph.node, params.name)
            if used_node.op_type in ('Conv', 'Gemm'):
                params_scale = config['scale']
            elif used_node.op_type in ('BatchNormalization'):
                params_scale = get_param_limit(onnx_model, used_node, float_data)
            else:
                params_scale = config['scale']
            '''
            if used_node.op_type in ('Gemm', 'BatchNormalization'):
                print("=================== {} ===================".format(params.name))
                print(float_data)
                print("max: {}, min: {}".format(max(float_data), min(float_data)))
            '''
            int_data_Q15 = _Q15(np.array(float_data) / params_scale, 'Parameter')

            if args.sparse:
                cols = []
                rows = []
                first_tile_index = []
            if args.sparse and (params.name in conv_param_names or params.name in gemm_param_names):
                # transform the sparse matrix into BSR format
                # layout: NCWHC
                node_idx = main_names.index(params.name)
                layer_config = model_config[node_idx]
                if params.name in conv_param_names:
                    data, cols, rows = toBSR(int_data_Q15, layer_config, params.dims, 'CONV')
                elif params.name in gemm_param_names:
                    data, cols, rows = toBSR(int_data_Q15, layer_config, params.dims, 'GEMM')
                    first_tile_index = find_first_tile_index(cols, rows, layer_config, params.dims, 'GEMM')
                int_data_Q15 = data

            data_len = len(int_data_Q15)
            assert data_len > 0
            slot = parameters_slot

            model_parameters_info.write(to_bytes(slot.offset, size=32))  # params_offset
            if Constants.param_bin:
                model_parameters_info.write(to_bytes(0, size=32))  # params_fram_offset
            model_parameters_info.write(to_bytes(data_len * 2, size=32))  # A _q15 is 16-bit
            # XXX: adjuct the length of cols and rows
            if args.sparse:
                model_parameters_info.write(to_bytes(slot.cols_offset, size=32))  # cols_offset
                model_parameters_info.write(to_bytes(slot.rows_offset, size=32))  # rows_offset
                model_parameters_info.write(to_bytes(slot.first_tile_index_offset, size=32))  # first_tile_index_offset
                if len(cols) == 0:
                    # +1 for bias
                    cols = rows = first_tile_index = [0]
                slot.cols.write(to_bytes(cols))
                slot.rows.write(to_bytes(rows))
                slot.first_tile_index.write(to_bytes(first_tile_index))
                slot.cols_offset += 2 * len(cols)
                slot.rows_offset += 2 * len(rows)
                slot.first_tile_index_offset += 2 * len(first_tile_index)

            slot.target.write(to_bytes(int_data_Q15))
            slot.offset += 2 * len(int_data_Q15)
            model_parameters_info.write(to_bytes(16, size=8)) # bitwidth
        elif params.data_type == onnx.TensorProto.INT64:
            if params.int64_data:
                int64_data = params.int64_data
            else:
                int64_data = decode_raw_data(params)
            data_len = len(int64_data)
            assert data_len > 0
            slot = parameters_slot
            model_parameters_info.write(to_bytes(slot.offset, size=32))  # params_offset
            if Constants.param_bin:
                model_parameters_info.write(to_bytes(0, size=32))  # params_fram_offset
            model_parameters_info.write(to_bytes(data_len * 8, size=32))
            if args.sparse:
                model_parameters_info.write(to_bytes(slot.cols_offset, size=32))  # cols_offset
                model_parameters_info.write(to_bytes(slot.rows_offset, size=32))  # rows_offset
                model_parameters_info.write(to_bytes(slot.first_tile_index_offset, size=32))  # first_tile_index_offset
            for param in int64_data:
                slot.target.write(to_bytes(param, size=64))
                slot.offset += 8
                if args.sparse:
                    if len(cols) == 0:
                        # +1 for bias
                        cols = rows = first_tile_index = [0]
                    slot.cols.write(to_bytes(cols))
                    slot.rows.write(to_bytes(rows))
                    slot.first_tile_index.write(to_bytes(first_tile_index))
                    slot.cols_offset += 2 * len(cols)
                    slot.rows_offset += 2 * len(rows)
                    slot.first_tile_index_offset += 2 * len(first_tile_index)
            model_parameters_info.write(to_bytes(64, size=8)) # bitwidth
        else:
            assert False
        model_parameters_info.write(to_bytes(slot.slot_id, size=8))  # slot
        if len(params.dims) == 4:
            channels = params.dims[1]
        else:
            channels = 0
        logger.info('dims = %r, length = %d', params.dims, data_len)
        for dim in params.dims:
            model_parameters_info.write(to_bytes(dim))
        # dims are always 4 uint16_t's in C++
        for _ in range(4 - len(params.dims)):
            model_parameters_info.write(to_bytes(0))
        logger.info("{} scale: {}".format(params.name, params_scale))
        write_scale(model_parameters_info, params_scale)

    # common to input and non-inputs
    model_parameters_info.write(to_bytes(0, size=8))                 # param_flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        model_parameters_info.write(to_bytes(0, size=8))             # extra_info
    model_parameters_info.write(to_bytes(parameter_info_idx))        # parameter_info_idx
    parameter_info_idx += 1

# Placeholder for ParameterInfo of intermediate values
intermediate_parameters_info = outputs['intermediate_parameters_info']
for idx, n in enumerate(nodes):
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_offset
    if Constants.param_bin:
        intermediate_parameters_info.write(to_bytes(0, size=32))  # params_fram_offset
    intermediate_parameters_info.write(to_bytes(0, size=32))  # params_len
    if args.sparse:
        intermediate_parameters_info.write(to_bytes(0, size=32))  # params_cols_offset
        intermediate_parameters_info.write(to_bytes(0, size=32))  # params_rows_offset
        intermediate_parameters_info.write(to_bytes(0, size=32))  # first_tile_index_offset
    intermediate_parameters_info.write(to_bytes(0, size=8))  # bitwidth
    intermediate_parameters_info.write(to_bytes(0, size=8))  # slot
    intermediate_parameters_info.write(to_bytes(0))         # dummy
    for _ in range(4):  # dims[4]
        intermediate_parameters_info.write(to_bytes(0))
    intermediate_parameters_info.write(to_bytes(0))   # scale
    intermediate_parameters_info.write(to_bytes(0, size=8))     # param_flags
    for _ in range(Constants.EXTRA_INFO_LEN):
        intermediate_parameters_info.write(to_bytes(0, size=8)) # extra_info
    intermediate_parameters_info.write(to_bytes(parameter_info_idx))             # parameter_info_idx
    parameter_info_idx += 1

def ensure_channel_last(images, data_layout):
    if data_layout in (DataLayout.NEUTRAL, DataLayout.NHWC, DataLayout.NWC):
        return images
    elif data_layout == DataLayout.NCW:
        return np.transpose(images, axes=(0, 2, 1))  # NCW => NWC
    elif data_layout == DataLayout.NCHW:
        return np.transpose(images, axes=(0, 2, 3, 1))  # NCHW => NHWC
    else:
        raise NotImplementedError

images = ensure_channel_last(model_data.images, model_data.data_layout)
for idx in range(model_data.images.shape[0]):
    im = images[idx, :]
    # load_data returns NCHW
    # https://stackoverflow.com/a/34794744

    int_data_Q15 = _Q15(im.flatten(order='C') / config['input_scale'], 'Input')
    outputs['samples'].write(to_bytes(int_data_Q15))
    if args.write_images:
        import cv2
        os.makedirs('images', exist_ok=True)
        # Restore conanical image format (H, W, C)
        im = np.squeeze(im * 256)
        cv2.imwrite(f'images/test{idx:02d}.png', im)

print("labels: ", model_data.labels)
for label in model_data.labels:
    outputs['labels'].write(to_bytes(label, size=8))

if args.write_images:
    with open('images/ans.txt', 'w') as f:
        f.write(' '.join(map(str, model_data.labels)))

pathlib.Path('build').mkdir(exist_ok=True)

# _parameters_
with open('build/data.cpp', 'w') as output_c, open('build/data.h', 'w') as output_h:
    output_h.write('''
#pragma once

#include <stdint.h>

struct ParameterInfo;
struct Model;
struct Node;

''')
    for item in itertools.chain(dir(Constants), config.keys()):
        if hasattr(Constants, item):
            if item.startswith('__'):
                continue
            val = getattr(Constants, item)
        else:
            val = config[item]
            if not isinstance(val, (int, float, np.int64, np.int32)):
                continue
        # Making it long to avoid overflow for expressions like
        # INTERMEDIATE_VALUES_SIZE * NUM_SLOTS on 16-bit systems
        suffix = 'l' if item == 'intermediate_values_size' else ''
        output_h.write(f'#define {item.upper()} ')
        if isinstance(val, str):
            output_h.write(f'"{val}"')
        elif isinstance(val, list):
            output_h.write('{' + ', '.join(map(str, val)) + '}')
        else:
            output_h.write(f'{val}')
        output_h.write(f'{suffix}\n')

    output_c.write('''
#include "data.h"
#include "cnn_common.h"
#include "platform.h"
''')

    # ops
    output_h.write('\n')
    for idx, op in enumerate(ops):
        output_h.write(f'#define Op{op} {idx}\n')

    for op in ops:
        output_h.write('void alloc_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);\n'.format(op.lower()))
        output_h.write('void handle_{}(struct Model *model, const struct ParameterInfo *input[], struct ParameterInfo *output, const struct Node* node);\n'.format(op.lower()))
    output_c.write('const handler handlers[] = {\n')
    for op in ops:
        output_c.write(f'    handle_{op},\n'.lower())
    output_c.write('};\n')
    output_c.write('const allocator allocators[] = {\n')
    for op in ops:
        output_c.write(f'    alloc_{op},\n'.lower())
    output_c.write('};\n')
    for op in ops:
        if op in inplace_update_ops:
            output_c.write(textwrap.dedent(f'''
                void alloc_{op.lower()}(struct Model *model, const struct ParameterInfo *[], struct ParameterInfo *output, const struct Node*) {{
                    SlotInfo *cur_slot_info = get_slot_info(model, output->slot);
                    if (cur_slot_info) {{
                        cur_slot_info->user = model->layer_idx;
                    }}
                }}
            '''))
        else:
            output_c.write(textwrap.dedent(f'''
                void __attribute__((weak)) alloc_{op.lower()}(struct Model *model, const struct ParameterInfo *[], struct ParameterInfo *output, const struct Node*) {{
                    ERROR_OCCURRED();
                }}
            '''))
        output_c.write(textwrap.dedent(f'''
            void __attribute__((weak)) handle_{op.lower()}(struct Model *model, const struct ParameterInfo *[], struct ParameterInfo *output, const struct Node*) {{
                ERROR_OCCURRED();
            }}
        '''))

    # data
    for idx, name in enumerate(other_flags):
        output_h.write(f'#define {name} {2**idx}\n')
    output_h.write(f'#ifdef __MSP430__ \n')
    output_h.write(f'#define DATA_SECTION_NVM _Pragma("DATA_SECTION(\\".nvm2\\")")\n')
    output_h.write(f'#else\n')
    output_h.write(f'#define DATA_SECTION_NVM\n')
    output_h.write(f'#endif\n')

    def hex_str(arr):
        return '  ' + ', '.join([f'0x{num:02x}' for num in arr]) + ',\n'

    def define_var(var_name, data):
        output_h.write(f'''
extern const uint8_t * const {var_name};
#define {var_name.upper()}_LEN {len(data)}
''')

        if Constants.param_bin and var_name[:-len('_data')] in EXTERNAL_DATA:
            return

        # #define with _Pragma seems to be broken :/
        output_c.write(f'''
DATA_SECTION_NVM const uint8_t _{var_name}[{len(data)}] = {{
''')
        n_pieces, remaining = divmod(len(data), 16)
        for idx in range(n_pieces):
            output_c.write(hex_str(data[idx*16:(idx+1)*16]))
        if remaining:
            output_c.write(hex_str(data[len(data) - remaining:len(data)]))
        output_c.write(f'''}};
const uint8_t * const {var_name} = _{var_name};
''')

    for var_name, data_obj in outputs.items():
        full_var_name = var_name + '_data'
        data_obj.seek(0)
        if full_var_name == 'samples_data':
            data = data_obj.read(2*config['total_sample_size'])
        else:
            data = data_obj.read()
        define_var(full_var_name, data)

with open('samples.bin', 'wb') as f:
    samples = outputs['samples']
    samples.seek(0)
    f.write(samples.read())

if Constants.param_bin:
    for var_name in EXTERNAL_DATA:
        with open(f'{var_name}.bin', 'wb') as f:
            data = outputs[var_name]
            data.seek(0)
            f.write(data.read())

