import onnx, onnx.numpy_helper
import argparse
import numpy as np
from scipy.sparse import csr_matrix

def printArgs(args):
    print('\n => Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('======================\n')
    return

def lowering(tensor, shape):
    origin_size = shape
    matrix = tensor.reshape(shape[0], -1)
    return matrix

def toBSR(matrix, dims, width):
    append_size = width - matrix.shape[1] % width
    if append_size != width:
        matrix = np.concatenate((matrix, np.zeros((len(matrix), append_size))), 1)
    bsr = csr_matrix(matrix).tobsr((1, width))
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
    if len(output_shape) == 4:
        print('cols: {}'.format(len(cols)))
        return len(cols) * output_shape[2] * output_shape[3]
    elif len(output_shape) == 2:
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model', action='store', default=None)
    parser.add_argument('--arch', action='store', default='LeNet_5')
    parser.add_argument('--prune', action='store', default='intermittent', help='Pruning methods: intermittent | energy')
    parser.add_argument('--group', action='store', type=int, default=5, help='Group size')
    args = parser.parse_args()
    printArgs(args)

    graph = []
    if args.prune == None:
        print('ERROR: Please choose the correct pruning strategies: {intermittent | energy}')
        exit()

    if args.prune == 'intermittent':
        pass
    elif args.prune == 'energy':
        pass

    if args.onnx_model == None:
        print('ERROR: Please choose the pruned model')
        exit()

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
    for idx, node in enumerate(nodes):
        shape = node.dims
        # print(shape)
        matrix = onnx.numpy_helper.to_array(node)
        if node.name in main_names:
            matrix = lowering(matrix, shape)
            matrix = toBSR(matrix, shape, args.group)
        sparse_node = {
            'dims': shape,
            'weights': matrix
        }
        graph.append(sparse_node)

    # getVal(graph[], 0)
    # printGroups(graph[2])
    output_shape = [[1,8,28,28], [1,20,14,14], [1, 256], [1, 10]]
    total_job = 0
    job = getJob(graph[0], output_shape[0])
    total_job += job
    job = getJob(graph[2], output_shape[1])
    total_job += job
    job = getJob(graph[4], output_shape[2])
    total_job += job
    job = getJob(graph[6], output_shape[3])
    total_job += job
    print('total_job: {}'.format(total_job))






