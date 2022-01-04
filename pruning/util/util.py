from __future__ import print_function
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix
from torchsummary import summary

cwd = os.getcwd()
sys.path.append(cwd+'/../')

def print_layer_info(model):
    index = 0
    print()
    for m in model.modules():
        if hasattr(m, 'alpha'):
            print('MaskLayer', index, ':',
                    m.alpha.data.nelement()-int(m.alpha.data.eq(1.0).sum()), 'of',
                    m.alpha.data.nelement(), 'is blocked')
            index += 1
    print()
    return

def print_args(args):
    print('\n==> Setting params:')
    for key in vars(args):
        print(key, ':', getattr(args, key))
    print('====================\n')
    return

def lowering(tensor, shape):
    origin_size = shape
    matrix = tensor.reshape(shape[0], -1)
    return matrix

def toBSR(matrix, group):
    bsr = csr_matrix(matrix).tobsr(group)
    return bsr

class beta_penalty():
    def __init__(self, model, penalty, lr, prune_target):
        self.penalty = float(penalty)
        self.lr = float(lr)
        self.penalty_target = []
        for m in model.modules():
            if isinstance(m, MaskLayer):
                if prune_target:
                    if (prune_target == 'ip') and (m.conv == False):
                        self.penalty_target.append(m.beta)
                    if (prune_target == 'conv') and (m.conv == True):
                        self.penalty_target.append(m.beta)
                else:
                    self.penalty_target.append(m.beta)
        return

    def update_learning_rate(self, lr):
        self.lr = float(lr)
        return

    def penalize(self):
        for index in range(len(self.penalty_target)):
            self.penalty_target[index].data.sub_(self.penalty*self.lr)
        return

def load_state(model, state_dict):
    param_dict = dict(model.named_parameters())
    state_dict_keys = state_dict.keys()
    cur_state_dict = model.state_dict()
    for key in cur_state_dict:
        if key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key])
        elif key.replace('module.','') in state_dict_keys:
            cur_state_dict[key].copy_(state_dict[key.replace('module.','')])
        elif 'module.'+key in state_dict_keys:
            cur_state_dict[key].copy_(state_dict['module.'+key])
    return

class dropout_update():
    def __init__(self, dropout_list, mask_list):
        self.dropout_list = dropout_list
        self.mask_list = mask_list
        self.dropout_param_list = []
        for i in range(len(self.dropout_list)):
            self.dropout_param_list.append(self.dropout_list[i].p)
        return

    def update(self):
        for i in range(len(self.dropout_list)):
            mask = self.mask_list[i].alpha.data
            dropout_tmp_value = float(mask.eq(1.0).sum()) / float(mask.nelement())
            dropout_tmp_value = dropout_tmp_value * self.dropout_param_list[i]
            self.dropout_list[i].p = dropout_tmp_value
        return

class Prune_Op():
    def __init__(self, model, threshold, struct, evaluate=False):
        width = struct[0] * struct[1] * struct[2] * struct[3]
        if not evaluate:
            model.weights_pruned = []
            node_idx = 0
            pruning_ratios = self.setPruningRatios(model)
            for m in model.modules():
               if isinstance(m, nn.Linear):
                    tmp_pruned = m.weight.data.clone()
                    append_size = width - tmp_pruned.shape[1]%width
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width)
                    shape = tmp_pruned.shape
                    tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx])
                    # tmp_pruned[:, -1] = 0
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1)
                    tmp_pruned = tmp_pruned[:, 0:m.weight.data.shape[1]]
                    model.weights_pruned.append(tmp_pruned)
                    node_idx += 1

               elif isinstance(m, nn.Conv2d):
                    tmp_pruned = m.weight.data.clone()
                    # tmp_pruned = tmp_pruned.permute(0, 2, 3, 1) # NCHW -> NHWC
                    original_size = tmp_pruned.size()
                    # Origin: im2col: [20, 1, 5, 5] -> [20, 25]
                    # Modified: [20, 5, 5, 1] -> [20, 25]
                    # tmp_pruned = tmp_pruned.reshape(original_size[0], -1)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    if tmp_pruned.shape[1] != width:
                        append_size = width - tmp_pruned.shape[1]%width
                        if append_size != width:
                            tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1) # Append: [20, 25] -> [20, 32]
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width) # Slice by width: [20,32] -> [20, 4, 8]
                    shape = tmp_pruned.shape
                    tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx])
                    # tmp_pruned[:, -1] = 0
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    tmp_pruned = tmp_pruned[:, 0:m.weight.data[0].nelement()]
                    tmp_pruned = tmp_pruned.view(original_size)
                    # tmp_pruned = tmp_pruned.permute(0, 3, 1, 2) # NHWC -> NCHW
                    model.weights_pruned.append(tmp_pruned)
                    node_idx += 1
            else:
                pass

        self.weights_pruned = model.weights_pruned
        self.model = model
        self.print_info()
        self.prune_weight()
        return

    def criteria(self, tmp_pruned, pruning_ratio):
        values = tmp_pruned.pow(2.0).mean(2, keepdim=True).pow(0.5)
        tmp_val = np.array(values.tolist()).flatten()
        nonzero_values = [val for val in tmp_val if val != 0]
        n_values = len(nonzero_values)
        threshold = sorted(nonzero_values)[:int(n_values * pruning_ratio)][-1]
        return values.expand(tmp_pruned.shape).lt(threshold)

    def prune_weight(self):
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data[self.weights_pruned[index]] = 0
                index += 1
            elif isinstance(m, nn.Conv2d):
                m.weight.data[self.weights_pruned[index]] = 0
                index += 1
        return

    def setPruningRatios(self, model):
        # https://gist.github.com/georgesung/ddb3a0b0412513d8811696293d8b1771
        # summary(model, (1, 28, 28), device='cpu')
        output_shapes = [[1, 8, 28, 28], \
                         [1, 16, 14, 14], \
                         [1, 275], \
                         [1, 25]]
                        # [1, 256]
                        # [1, 10]
        pruning_ratios = [0] * len(output_shapes)
        jobs = []
        node_idx = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                job = self.getJobs(m, (1, 25), output_shapes, node_idx)
                jobs.append(job)
                node_idx += 1

        # adjusted by manually
        candidates = [0.1, 0.2, 0.3, 0.4]
        order = sorted(range(len(jobs)), key=lambda k : jobs[k])
        for i in range(len(order)):
            pruning_ratios[order[i]] = candidates[i]
        print('pruning_ratios: {}'.format(pruning_ratios))
        return pruning_ratios


    def getJobs(self, node, group_size, output_shapes, node_idx):
        width = group_size[1]
        shape = node.weight.data.shape
        matrix = node.weight.data
        matrix = matrix.reshape(shape[0], -1)
        append_size = width - matrix.shape[1]%width
        matrix = torch.cat((matrix, matrix[:, 0:append_size]), 1)
        bsr = csr_matrix(matrix).tobsr(group_size)
        data = bsr.data
        cols = bsr.indices
        rows = bsr.indptr
        if isinstance(node, nn.Linear):
            job = len(cols)
        elif isinstance(node, nn.Conv2d):
            job = len(cols) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
        return job

    def getReuse(self):
        pass

    def print_info(self):
        print('------------------------------------------------------------------')
        print('- Intermittent-aware weight pruning info:')
        pruned_acc = 0
        total_acc = 0
        for i in range(len(self.weights_pruned)):
            pruned = int(self.weights_pruned[i].sum())
            total = int(self.weights_pruned[i].nelement())
            pruned_acc += pruned
            total_acc += total
            print('- Layer '+str(i)+': '+'{0:10d}'.format(pruned)+' / '+\
                    '{0:10d}'.format(total)+ ' ('\
                    '{0:4.1f}%'.format(float(pruned)/total * 100.0)+\
                    ') weights are pruned')
        print('- Total  : '+'{0:10d}'.format(pruned_acc)+' / '+\
                '{0:10d}'.format(total_acc)+ ' ('\
                '{0:4.1f}%'.format(float(pruned_acc)/total_acc * 100.0)+\
                ') weights are pruned')
        print('------------------------------------------------------------------\n')
        return
