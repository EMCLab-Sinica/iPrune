from __future__ import print_function
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from scipy.sparse import csr_matrix
from torchsummary import summary
from torch.autograd import Variable
from itertools import chain
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd+'/../')

# https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
def activation_shapes(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    names = []
    shapes = []
    valid_layer = ("Conv2d", "Linear")
    for layer in summary:
        if layer[:layer.find('-')] in valid_layer:
            names.append(layer)
            shapes.append(summary[layer]["output_shape"])

    return names, shapes

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
    def __init__(self, model, train_loader, criterion, input_shape,args, evaluate=False):
        self.width = math.prod(args.group)
        self.train_loader = train_loader
        self.criterion = criterion
        self.args = args
        names, output_shapes = activation_shapes(model, input_shape)
        if not evaluate:
            if args.with_sen:
                pruning_ratios, sorted_idx, first_unpruned_idx = self.setPruningRatios2(model, args.pruned_ratios)
            else:
                pruned_ratios = self.getPrunedRatios(model.weights_pruned, output_shapes)
                pruning_ratios = self.setPruningRatios(model, args.candidates_pruning_ratios, output_shapes)

            print('pruned ratios: {}'.format(pruned_ratios))
            model.weights_pruned = []
            node_idx = 0
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    tmp_pruned = m.weight.data.clone()
                    append_size = self.width - tmp_pruned.shape[1]%self.width
                    tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, self.width)
                    shape = tmp_pruned.shape
                    if args.with_sen:
                        tmp_pruned = self.criteria2(tmp_pruned, sorted_idx[node_idx], first_unpruned_idx[node_idx])
                    else:
                        tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx], pruned_ratios[node_idx])
                    # tmp_pruned[:, -1] = 0
                    tmp_pruned = tmp_pruned.reshape(tmp_pruned.shape[0], -1)
                    tmp_pruned = tmp_pruned[:, 0:m.weight.data.shape[1]]
                    model.weights_pruned.append(tmp_pruned)
                    node_idx += 1

                elif isinstance(m, nn.Conv2d):
                    print(m)
                    tmp_pruned = m.weight.data.clone()
                    # tmp_pruned = tmp_pruned.permute(0, 2, 3, 1) # NCHW -> NHWC
                    original_size = tmp_pruned.size()
                    # Origin: im2col: [20, 1, 5, 5] -> [20, 25]
                    # Modified: [20, 5, 5, 1] -> [20, 25]
                    # tmp_pruned = tmp_pruned.reshape(original_size[0], -1)
                    tmp_pruned = tmp_pruned.view(original_size[0], -1)
                    if tmp_pruned.shape[1] != self.width:
                        append_size = self.width - tmp_pruned.shape[1]%self.width
                        if append_size != self.width:
                            tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1) # Append: [20, 25] -> [20, 32]
                    tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, self.width) # Slice by width: [20,32] -> [20, 4, 8]
                    shape = tmp_pruned.shape
                    if args.with_sen:
                        tmp_pruned = self.criteria2(tmp_pruned, sorted_idx[node_idx], first_unpruned_idx[node_idx])
                    else:
                        tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx], pruned_ratios[node_idx])

                    # tmp_pruned[:, -1] = 0
                    tmp_pruned = tmp_pruned.reshape(original_size[0], -1)
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

    def criteria(self, tmp_pruned, pruning_ratio, pruned_ratio):
        values = tmp_pruned.pow(2.0).mean(2, keepdim=True).pow(0.5)
        tmp_val = np.array(values.tolist()).flatten()
        n_values = len(tmp_val)
        pruning_ratio = 1 - float(1 - pruning_ratio) * float(1 - pruned_ratio)
        sorted_idx = sorted(range(len(tmp_val)), key = lambda k : tmp_val[k])[: int(n_values * pruning_ratio)]
        mask = np.zeros(tmp_val.shape, dtype=bool)
        mask[sorted_idx] = True
        mask = np.reshape(mask, values.shape)
        return torch.tensor(mask).expand(tmp_pruned.shape)

    def criteria2(self, tmp_pruned, sorted_idx, first_unpruned_idx):
        if first_unpruned_idx != -1:
            masked_indices = sorted_idx[:first_unpruned_idx]
        else:
            masked_indices = sorted_idx
        values = tmp_pruned.pow(2.0).mean(2, keepdim=True).pow(0.5)
        tmp_val = np.array(values.tolist()).flatten()
        mask = np.zeros(tmp_val.shape, dtype=bool)
        mask[masked_indices] = True
        mask = np.reshape(mask, values.shape)
        return torch.tensor(mask).expand(tmp_pruned.shape)

    def getPrunedRatios(self, masks, output_shapes):
        if masks == None or masks == []:
            return [0] * len(output_shapes)
        pruned_ratios = []
        for i in range(len(masks)):
            n_pruned = int(masks[i].sum())
            total = int(masks[i].nelement())
            pruned_ratios.append(float(n_pruned / total))
        return pruned_ratios


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

    def setPruningRatios(self, model, candidates, output_shapes):
        # https://gist.github.com/georgesung/ddb3a0b0412513d8811696293d8b1771
        pruning_ratios = [0] * len(output_shapes)
        metrics = []
        node_idx = 0
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if self.args.prune == 'intermittent':
                    metric = self.getJobs(m, (1, self.width), output_shapes, node_idx)
                    metrics.append(metric)
                elif self.args.prune == 'energy':
                    metric = self.getReuse(m, (1, self.width), output_shapes, node_idx)
                    metrics.append(metric)
                node_idx += 1
        print(output_shapes)
        print("Metrics : {}".format(metrics))
        order = sorted(range(len(metrics)), key=lambda k : metrics[k])
        print("Pruning Order: {}".format(order))
        for i in range(len(order)):
            pruning_ratios[order[i]] = candidates[i]
        print('pruning_ratios: {}'.format(pruning_ratios))
        return pruning_ratios

    def setPruningRatios2(self, model, total_pruning_ratio = 0.5):
        weights = []
        org_layers = []
        for m in model.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weights.append(m.weight)
                org_layers.append(m)

        def getGrad(train_loader, model, criterion, weights):
            model.eval()
            grad_one = [torch.zeros(w.size()) for w in weights]
            for batch_idx, (inputs, target) in enumerate(train_loader):
                if self.args.cuda:
                    inputs, target = inputs.cuda(), target.cuda()
                inputs, target = Variable(inputs), Variable(target)
                inputs.requires_grad_(True)
                output = model(inputs)
                loss = criterion(output, target)
                grad_params_1 = torch.autograd.grad(loss, weights)

                for j, gp in enumerate(grad_params_1):
                    grad_one[j] += gp

            grad_one = [g / len(train_loader) for g in grad_one]
            return grad_one

        grads = getGrad(self.train_loader, model, self.criterion, weights)

        def getI(model, org_layers, grads):
            res = []
            for i in range(len(org_layers)):
                res.append(torch.pow(grads[i] * org_layers[i].weight, 2))
            return res

        I = getI(model, org_layers, grads)

        group_size = [1, 1, 1, 5]

        def grouping(I, group = [1, 1, 1, 5]): # group size: one row
            I_groups = []
            for i in range(len(I)):
                layer = I[i]
                layer = layer.reshape(-1).data
                layer = np.add.reduceat(layer, np.arange(0, len(layer), group[3]))
                I_groups.append(layer)
            return I_groups

        I_groups = grouping(I, group_size)

        metrics_sort = []
        for i in range(len(I_groups)):
            layer = I_groups[i]
            metric_sort = sorted(range(len(layer)), key=lambda idx : layer[idx])
            metrics_sort.append(metric_sort)

        input_sizes = [[1,1,28,28], [1,8,14,14], [16*4*4, 256], [256, 10]]
        output_sizes = []
        def getJob(model, group_size, input_sizes):
            idx = 0
            jobs = []
            total_job = 0
            per_group_job = []
            for m in model.children():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    if len(m.weight.shape) == 4:
                        input_size = input_sizes[idx]
                        kernel_size = m.weight.shape[2:]
                        padding = m.padding
                        stride = m.stride
                        out_h = int((input_size[2] + 2 * padding[0] - kernel_size[0])/stride[0] + 1)
                        out_w = int((input_size[3] + 2 * padding[1] - kernel_size[1])/stride[1] + 1)
                        output_sizes.append([out_h, out_w])
                        per_group_job.append(out_h * out_w)
                        n_weights = np.prod(m.weight.shape)
                        job = int(n_weights * out_h * out_w / np.prod(group_size))
                        jobs.append(job)
                        total_job += job
                        idx += 1
                    elif len(m.weight.shape) == 2:
                        n_weights = np.prod(m.weight.shape)
                        output_sizes.append([1, m.weight.shape[0]])
                        job = int(n_weights / np.prod(group_size))
                        jobs.append(job)
                        per_group_job.append(1)
                        total_job += job
                        idx += 1
            return jobs, total_job, per_group_job

        jobs, total_job, per_group_job = getJob(model, group_size, input_sizes)
        print('jobs: {}'.format(jobs))
        print('total job: {}'.format(total_job))
        print('output size: {}'.format(output_sizes))

        acc_I = []
        percentages = []
        for i in range(len(metrics_sort)):
            acc = np.array(I_groups[i])[metrics_sort[i]]
            for j in range(len(acc) - 1):
                acc[j + 1] += acc[j]
            acc /= per_group_job[i]
            percentage = np.linspace(0, 100, len(acc))
            percentages.append(percentage)
            acc_I.append(acc)

        def getPruningRatio(acc_I, total_pruning_ratio, total_job, jobs, per_group_job, threshold = 0):
            total_acc = list(chain.from_iterable(acc_I))
            total_acc.sort()
            PR = []
            indices = []
            remain_job = total_job
            ideal_job = int(total_job * (1 - total_pruning_ratio))
            for threshold in total_acc:
                PR = []
                indices = []
                for i in range(len(acc_I)):
                    iters = iter(idx for idx, val in enumerate(acc_I[i]) if val > threshold)
                    first_greater_than_idx = next(iters, -1)
                    if first_greater_than_idx == -1:
                        print('Layer {} is exhausted iteration'.format(i + 1))
                        first_greater_than_idx = len(acc_I[i])
                    PR.append(first_greater_than_idx / len(acc_I[i]) * 100)
                    indices.append(first_greater_than_idx)
                remain_job = int(sum(jobs - np.array(indices) * np.array(per_group_job)))
                if(remain_job <= ideal_job):
                    break

            print('Remain: {}'.format(remain_job))
            print('Ideal remain job: {}'.format(ideal_job))
            print('threshold: {}'.format(threshold))
            return PR, indices

        print('============================================================================================')
        PR, indices = getPruningRatio(acc_I, total_pruning_ratio, total_job, jobs, per_group_job)
        print('Total Prune: {}'.format(total_pruning_ratio))
        print('Pruning Ratio: {}'.format(PR))
        return PR, metrics_sort, indices

    def getJobs(self, node, group_size, output_shapes, node_idx):
        shape = node.weight.data.shape
        matrix = node.weight.data
        matrix = matrix.reshape(shape[0], -1)
        append_size = self.width - matrix.shape[1]%self.width
        matrix = torch.cat((matrix, matrix[:, 0:append_size]), 1)
        matrix = matrix.cpu()
        bsr = csr_matrix(matrix).tobsr(group_size)
        data = bsr.data
        cols = bsr.indices
        rows = bsr.indptr
        if isinstance(node, nn.Linear):
            job = len(cols)
        elif isinstance(node, nn.Conv2d):
            job = len(cols) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
        return job

    def getReuse(self, node, group_size, output_shapes, node_idx):
        shape = node.weight.data.shape
        matrix = node.weight.data
        matrix = matrix.reshape(shape[0], -1)
        append_size = self.width - matrix.shape[1]%self.width
        matrix = torch.cat((matrix, matrix[:, 0:append_size]), 1)
        bsr = csr_matrix(matrix).tobsr(group_size)
        data = bsr.data
        cols = bsr.indices
        rows = bsr.indptr
        if isinstance(node, nn.Linear):
            weight_reuse = len(cols) * self.width
            input_reuse = weight_reuse
            output_reuse = len(cols)
            psum_read = 2 * len(cols)
            psum_write = len(cols)
            reuse = weight_reuse + input_reuse + output_reuse + psum_read + psum_write
        elif isinstance(node, nn.Conv2d):
            weight_reuse = self.width * len(cols) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
            input_reuse = weight_reuse
            output_reuse = len(cols) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
            psum_read = 2 * (len(cols) - 1) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
            psum_write = (len(cols) - 1) * output_shapes[node_idx][2] * output_shapes[node_idx][3]
            reuse = weight_reuse + input_reuse + output_reuse + psum_read + psum_write
        return reuse


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
