from __future__ import print_function
import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import math
import models
import copy
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from scipy.sparse import csr_matrix
from torchsummary import summary
from torch.autograd import Variable
from itertools import chain
from collections import OrderedDict
from config import config
from tqdm import tqdm, trange

cwd = os.getcwd()
sys.path.append(cwd+'/../')

def set_logger(level=None):
    # Set logger info
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s',
        datefmt='%Y%m%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

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

def to_onnx(source, name, args):
    if args.arch == 'LeNet_5':
        model = models.LeNet_5(None)
        input_shape = (1,28,28)
        dummy_input = Variable(torch.randn(1, 1, 28, 28))
    elif args.arch == 'SqueezeNet':
        model = models.SqueezeNet(None)
        input_shape = (1,32,32)
        dummy_input = Variable(torch.randn(1, 3, 32, 32))
    # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379/14
    state_dict = torch.load(source)['state_dict']
    model.load_state_dict(state_dict)
    converted_name = "./Intermittent_Aware/onnx_models/{}.onnx".format(args.arch)
    torch.onnx.export(model, dummy_input, converted_name)
    print('Converted model: {}'.format(converted_name))
    return

def open():
    pass

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

def nchw2nhwc(arr):
    # print(arr.shape)
    return arr.permute(0, 2, 3, 1) # NCHW -> NHWC

def getJobs(args, node, group_size, output_shapes, node_idx):
    width = math.prod(group_size)
    shape = node.weight.data.shape
    matrix = node.weight.data
    if args.layout == 'nhwc' and isinstance(node, nn.Conv2d):
        matrix = nchw2nhwc(matrix)
    matrix = matrix.reshape(shape[0], -1)
    append_size = width - matrix.shape[1]%width
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

def getReuse(args, node, group_size, output_shapes, node_idx):
    layer_config = config[args.arch][node_idx]
    width = np.prod(args.group)
    nvm_access_cost = 100
    vm_access_cost = 1

    shape = node.weight.data.shape
    matrix = node.weight.data
    matrix = matrix.reshape(shape[0], -1)
    append_size = width - matrix.shape[1] % width
    if append_size == width:
        append_size = 0
    matrix = torch.cat((matrix, matrix[:, 0:append_size]), 1).cpu()
    bsr = csr_matrix(matrix).tobsr(group_size)
    data = bsr.data
    cols = bsr.indices
    rows = bsr.indptr

    nvm_jobs = 0
    nvm_read_weights = 0
    nvm_read_inputs = 0
    vm_jobs = 0
    vm_read_psum = 0
    vm_write_psum = 0
    vm_read_weights = 0
    vm_read_inputs = 0
    if isinstance(node, nn.Linear):
        logger_.debug("Node: {}".format(node_idx))
        # input stationary
        nvm_read_inputs += layer_config['input'][2]
        for i in range(1, len(rows)):
            if rows[i] - rows[i - 1] != 0:
                # TODO: indexing cost
                nvm_read_weights += (rows[i] - rows[i - 1]) * width
                # XXX: All channels can't be loaded in a weight tile
                nvm_jobs += layer_config['output'][0] * layer_config['output'][1]
                vm_jobs += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_psum += 2 * ((rows[i] - rows[i - 1]) - 1) * layer_config['output'][0] * layer_config['output'][1]
                vm_write_psum += ((rows[i] - rows[i - 1]) - 1) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_weights += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_inputs += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]

    elif isinstance(node, nn.Conv2d):
        # weight stationary
        n_input_nvm_access = min(math.ceil((len(cols) * width) / math.prod(layer_config['tile']['weight'])), layer_config['filter'][3] / layer_config['tile']['weight'][3])
        logger_.debug("Node: {}".format(node_idx))
        logger_.debug("n_nvm_input: {}".format(n_input_nvm_access))
        # nvm_access_per_ifm = (layer_config['filter'][3] / layer_config['tile']['weight'][3]) * math.ceil(layer_config['tile']['weight'][1]/layer_config['stride'])
        nvm_access_per_ifm = n_input_nvm_access * math.ceil(layer_config['tile']['weight'][1]/layer_config['stride'])
        nvm_read_inputs += math.prod(layer_config['input']) * nvm_access_per_ifm
        for i in range(1, len(rows)):
            if rows[i] != rows[i - 1]:
                # TODO: indexing cost
                nvm_read_weights += (rows[i] - rows[i - 1]) * width
                # XXX: All channels can't be loaded in a wieght tile
                nvm_jobs += layer_config['output'][0] * layer_config['output'][1]
                vm_jobs += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_psum += 2 * ((rows[i] - rows[i - 1]) - 1) * layer_config['output'][0] * layer_config['output'][1]
                vm_write_psum += ((rows[i] - rows[i - 1]) - 1) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_weights += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]
                vm_read_inputs += (rows[i] - rows[i - 1]) * layer_config['output'][0] * layer_config['output'][1]
    logger_.debug('nvm_read_weights: {}'.format(nvm_read_weights))
    logger_.debug('nvm_read_inputs: {}'.format(nvm_read_inputs))
    logger_.debug('nvm_jobs: {}'.format(nvm_jobs))
    logger_.debug('vm_jobs: {}'.format(vm_jobs))
    vm_access = (vm_jobs + vm_read_inputs + vm_read_weights + vm_read_psum + vm_write_psum)
    nvm_access = (nvm_jobs + nvm_read_weights + nvm_read_inputs)
    return vm_access * vm_access_cost + nvm_access * nvm_access_cost, (vm_access, nvm_access)

def prune_weight_layer(m, mask):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.data[mask] = 0
    return m.weight.data

def prune_weight(model):
    logger_.info('Start pruning weight...')
    index = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data[model.weights_pruned[index]] = 0
            index += 1
        elif isinstance(m, nn.Conv2d):
            m.weight.data[model.weights_pruned[index]] = 0
            index += 1
    logger_.info('Finish pruning weight.')
    return model

class SimulatedAnnealing():
    def __init__(self, model, start_temp, stop_temp, cool_down_rate, perturbation_magnitude, target_sparsity, args, evaluate_function, input_shape, output_shapes, mask_maker):
        self.model_ = model
        self.start_temp_ = start_temp
        self.stop_temp_ = stop_temp
        self.cool_down_rate_ = cool_down_rate
        self.perturbation_magnitude_ = perturbation_magnitude
        self.target_sparsity_ = target_sparsity
        self.args_ = args
        self.evaluator_ = evaluate_function

        self.sparsities_ = None

        self.cur_perf_ = -np.inf
        self.best_perf_ = -np.inf
        self.best_sparsities_ = []
        self.search_history_ = []

        self.input_shape_ = input_shape
        self.output_shapes_ = output_shapes
        self.mask_maker_ = mask_maker
        self.start()

    def get_n_node(self):
        cnt = 0
        for m in self.model_.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                cnt += 1
        return cnt

    def get_sparsities(self):
        return self.best_sparsities_

    def rescale_sparsities(self, sparsities, target_sparsity):
        metrics = []
        node_idx = 0
        for m in self.model_.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if self.args_.prune == 'intermittent':
                    metric = getJobs(self.args_, m, (1, np.prod(self.args_.group)), self.output_shapes_, node_idx)
                    metrics.append(metric)
                elif self.args_.prune == 'energy':
                    metric, (vm_access, nvm_access) = getReuse(self.args_, m, (1, np.prod(self.args_.group)), self.output_shapes_, node_idx)
                    metrics.append(metric)
                node_idx += 1
        sparsities = sorted(sparsities)
        order = sorted(range(len(metrics)), key=lambda k : metrics[k])
        total_weight = 0
        total_weight_pruned = 0
        for i in range(len(sparsities)):
            node_idx = order[i]
            node_weights = self.model_.weights_pruned[node_idx]
            n_node_weights = np.prod(node_weights.shape)
            total_weight += n_node_weights
            total_weight_pruned += int(n_node_weights * sparsities[i])
        scale = target_sparsity / (total_weight_pruned / total_weight)
        sparsities = np.asarray(sparsities) * scale
        # check the result of rescalling
        total_weight_pruned_test = 0
        for i in range(len(sparsities)):
            node_idx = order[i]
            node_weights = self.model_.weights_pruned[node_idx]
            n_node_weights = np.prod(node_weights.shape)
            total_weight_pruned_test += int(n_node_weights * sparsities[i])
        logger_.debug('Metrics: {}'.format(metrics))
        logger_.debug('Rescale_sparsity: %s', total_weight_pruned_test / total_weight)

        return sparsities

    def init_sparsities(self):
        while True:
            sparsities = sorted(np.random.uniform(0, 1, self.get_n_node()))
            sparsities = self.rescale_sparsities(sparsities, target_sparsity=self.target_sparsity_)

            if sparsities is not None and sparsities[0] >= 0 and sparsities[-1] < 1:
                logger_.info('Gen sparsities: {}'.format(sparsities))
                self.sparsities_ = sparsities
                break

    def generate_perturbations(self):
        '''
        Generate perturbation to the current sparsities distribution.
        Returns:
        --------
        list
            perturbated sparsities
        '''
        logger_.info("Generating perturbations to the current sparsities...")

        # decrease magnitude with current temperature
        magnitude = self.cur_temp_ / self.start_temp_ * self.perturbation_magnitude_
        logger_.info('current perturation magnitude:%s', magnitude)

        while True:
            perturbation = np.random.uniform(-magnitude, magnitude, self.get_n_node())
            sparsities = np.clip(0, self.sparsities_ + perturbation, None)
            logger_.debug("sparsities before rescalling: {}".format(sparsities))

            sparsities = self.rescale_sparsities(sparsities, target_sparsity=self.target_sparsity_)
            logger_.debug("sparsities after rescalling: {}".format(sparsities))

            if sparsities is not None and sparsities[0] >= 0 and sparsities[-1] < 1:
                logger_.info("Sparsities perturbated:%s", sparsities)
                return sparsities

    def apply_sparsities(self, sparsities):
        pruning_ratios = [0] * self.get_n_node()
        metrics = []
        node_idx = 0
        for m in self.model_.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if self.args_.prune == 'intermittent':
                    metric = getJobs(self.args_, m, (1, np.prod(self.args_.group)), self.output_shapes_, node_idx)
                    metrics.append(metric)
                elif self.args_.prune == 'energy':
                    metric, (vm_access, nvm_access) = getReuse(self.args_, m, (1, np.prod(self.args_.group)), self.output_shapes_, node_idx)
                    metrics.append(metric)
                node_idx += 1
        sparsities = sorted(sparsities)
        order = sorted(range(len(metrics)), key=lambda k : metrics[k])
        for i in range(len(order)):
            pruning_ratios[order[i]] = sparsities[i]
        logger_.debug('Pruning order: {}'.format(order))
        logger_.debug('sparsities: {}'.format(sparsities))
        logger_.debug('Pruning ratios: {}'.format(pruning_ratios))
        return pruning_ratios

    def start(self):
        logger_.info('Starting Simulated Annealing...')
        it = 0
        self.init_sparsities()

        self.cur_temp_ = self.start_temp_
        while self.cur_temp_ > self.stop_temp_:
            logger_.info('Iter {}:'.format(it))
            logger_.info('Current Temperature: {}'.format(self.cur_temp_))

            while True:
                # generate perturbation
                model_masked = copy.deepcopy(self.model_)
                sparsities_perturbated = self.generate_perturbations()
                config_list = self.apply_sparsities(sparsities_perturbated)
                logger_.info('config_list for Pruner generated: {}'.format(config_list))

                # fast evaluation
                model_masked.weights_pruned = self.mask_maker_.get_masks(config_list)
                model_masked = prune_weight(model_masked)
                evaluation_result = self.evaluator_(model_masked)

                self.search_history_.append(
                    {'sparsity': self.sparsities_, 'performance': evaluation_result, 'pruning_ratios': config_list})

                evaluation_result *= -1

                # if better evaluation result, then accept the perturbation
                if evaluation_result > self.cur_perf_:
                    self.cur_perf_ = evaluation_result
                    self.sparsities_ = sparsities_perturbated

                    # save best performance and best params
                    if evaluation_result > self.best_perf_:
                        logger_.info('updating best model...')
                        self.best_perf_ = evaluation_result
                        self.best_sparsities_ = config_list

                        # save the overall best masked model
                        self.bound_model = model_masked
                    break
                # if not, accept with probability e^(-deltaE/current_temperature)
                else:
                    delta_E = np.abs(evaluation_result - self.cur_perf_)
                    probability = math.exp(-1 * delta_E /
                                           self.cur_temp_)
                    if np.random.uniform(0, 1) < probability:
                        self.cur_perf = evaluation_result
                        self.sparsities_ = sparsities_perturbated
                        break
            # cool down
            self.cur_temp_ *= self.cool_down_rate_
            it += 1
        logger_.info('Finish simulating anealing.')
        # print('History: ',self.search_history_)

class MaskMaker():
    def __init__(self, model, args, input_shape):
        self.args_ = args
        self.model_ = model
        self.input_shape = input_shape
        _, self.output_shapes_ = activation_shapes(model, input_shape)
        self.weight_masks_ = []
        self.pruned_ratios_ = self.getPrunedRatios(model.weights_pruned, self.output_shapes_)
        self.generate_masks()

    def getPrunedRatios(self, masks, output_shapes):
        if masks == None or masks == []:
            return [0] * len(output_shapes)
        pruned_ratios = []
        for i in range(len(masks)):
            n_pruned = int(masks[i].sum())
            total = int(masks[i].nelement())
            pruned_ratios.append(float(n_pruned / total))
        logger_.debug('Pruned ratios: {}'.format(pruned_ratios))
        return pruned_ratios

    def setPruningRatios(self, model, candidates, output_shapes):
        # https://gist.github.com/georgesung/ddb3a0b0412513d8811696293d8b1771
        pruning_ratios = [0] * len(output_shapes)
        metrics = []
        node_idx = 0
        width = np.prod(self.args_.group)
        for m in model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if self.args_.prune == 'intermittent':
                    metric = getJobs(self.args_, m, (1, width), output_shapes, node_idx)
                    metrics.append(metric)
                elif self.args_.prune == 'energy':
                    metric, (vm_access, nvm_access) = getReuse(self.args_, m, (1, width), output_shapes, node_idx)
                    metrics.append(metric)
                node_idx += 1
        order = sorted(range(len(metrics)), key=lambda k : metrics[k])
        for i in range(len(order)):
            pruning_ratios[order[i]] = candidates[i]
        logger_.debug("Metrics : {}".format(metrics))
        logger_.debug("Pruning Order: {}".format(order))
        logger_.debug('pruning_ratios: {}'.format(pruning_ratios))
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
                if self.args_.cuda:
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
        logger_.debug('jobs: {}'.format(jobs))
        logger_.debug('total job: {}'.format(total_job))
        logger_.debug('output size: {}'.format(output_sizes))

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
                        logger_.debug('Layer {} is exhausted iteration'.format(i + 1))
                        first_greater_than_idx = len(acc_I[i])
                    PR.append(first_greater_than_idx / len(acc_I[i]) * 100)
                    indices.append(first_greater_than_idx)
                remain_job = int(sum(jobs - np.array(indices) * np.array(per_group_job)))
                if(remain_job <= ideal_job):
                    break

            logger_.debug('Remain: {}'.format(remain_job))
            logger_.debug('Ideal remain job: {}'.format(ideal_job))
            logger_.debug('threshold: {}'.format(threshold))
            return PR, indices

        PR, indices = getPruningRatio(acc_I, total_pruning_ratio, total_job, jobs, per_group_job)
        logger_.debug('Total Prune: {}'.format(total_pruning_ratio))
        logger_.debug('Pruning Ratio: {}'.format(PR))
        return PR, metrics_sort, indices

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

    def generate_masks(self):
        names, output_shapes = activation_shapes(self.model_, self.input_shape)
        if self.args_.with_sen:
            pruning_ratios, sorted_idx, first_unpruned_idx = self.setPruningRatios2(self.model_, self.args_.pruned_ratios)
            self.generate_masks_(pruning_ratios)
        else:
            pruned_ratios = self.pruned_ratios_
            pruning_ratios = self.setPruningRatios(self.model_, self.args_.candidates_pruning_ratios, output_shapes)
            self.generate_masks_(pruning_ratios, pruned_ratios)


    def generate_masks_(self, pruning_ratios, pruned_ratios=[]):
        logger_.info('Start generating masks ...')
        node_idx = 0
        width = np.prod(self.args_.group)
        self.weight_masks_ = []
        if len(pruned_ratios) == 0:
            pruned_ratios = [0] * len(self.model_.weights_pruned)
        for m in self.model_.modules():
            if isinstance(m, nn.Linear):
                tmp_pruned = m.weight.data.clone()
                append_size = width - tmp_pruned.shape[1] % width
                tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1)
                tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width)
                shape = tmp_pruned.shape
                if self.args_.with_sen:
                    tmp_pruned = self.criteria2(tmp_pruned, sorted_idx[node_idx], first_unpruned_idx[node_idx])
                else:
                    tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx], pruned_ratios[node_idx])
                tmp_pruned = tmp_pruned.reshape(tmp_pruned.shape[0], -1)
                tmp_pruned = tmp_pruned[:, 0:m.weight.data.shape[1]]
                self.weight_masks_.append(tmp_pruned)
                node_idx += 1

            elif isinstance(m, nn.Conv2d):
                tmp_pruned = m.weight.data.clone()
                if self.args_.layout == 'nhwc':
                    tmp_pruned = tmp_pruned.permute(0, 2, 3, 1) # NCHW -> NHWC
                original_size = tmp_pruned.size()
                # Origin: im2col: [8, 1, 5, 5] -> [8, 25]
                # Modified: [8, 5, 5, 1] -> [8, 25]
                # For NHWC
                if self.args_.layout == 'nhwc':
                    if original_size[-1] % 2:
                        # pad: [8, 5, 5, 1] -> [8, 5, 5, 2]
                        padding_shape = (tmp_pruned.shape[0], tmp_pruned.shape[1], tmp_pruned.shape[2], 1)
                        padding_zeros = torch.zeros(padding_shape)
                        tmp_pruned = torch.cat((tmp_pruned.cpu(), padding_zeros), 3)
                tmp_pruned = tmp_pruned.reshape(original_size[0], -1) # [8, 5, 5, 2] -> [8, 50]
                # tmp_pruned = tmp_pruned.view(original_size[0], -1)
                if tmp_pruned.shape[1] != width:
                    append_size = width - tmp_pruned.shape[1] % width
                    if append_size != width:
                        tmp_pruned = torch.cat((tmp_pruned, tmp_pruned[:, 0:append_size]), 1) # Append: [8, 50] -> [8, 50]
                tmp_pruned = tmp_pruned.view(tmp_pruned.shape[0], -1, width) # Slice by width: [8,50] -> [8, 25, 2]
                shape = tmp_pruned.shape
                if self.args_.with_sen:
                    tmp_pruned = self.criteria2(tmp_pruned, sorted_idx[node_idx], first_unpruned_idx[node_idx])
                else:
                    tmp_pruned = self.criteria(tmp_pruned, pruning_ratios[node_idx], pruned_ratios[node_idx])

                if self.args_.layout == 'nhwc':
                    if original_size[-1] % 2:
                        tmp_pruned = torch.narrow(tmp_pruned, 2, 0, 1)
                tmp_pruned = tmp_pruned.reshape(original_size[0], -1)
                tmp_pruned = tmp_pruned[:, 0:m.weight.data[0].nelement()]
                tmp_pruned = tmp_pruned.view(original_size)
                # For NHWC
                if self.args_.layout == 'nhwc':
                    tmp_pruned = tmp_pruned.permute(0, 3, 1, 2) # NHWC -> NCHW
                self.weight_masks_.append(tmp_pruned)
                node_idx += 1
        logger_.info('Finish generating masks.')

    def get_masks(self, sparsities_perturbated=[]):
        if len(sparsities_perturbated) == 0:
            logger_.info('Get the mask without perturbating.')
            return self.weight_masks_
        logger_.info('Get the mask with perturbating.')
        self.generate_masks_(sparsities_perturbated, self.pruned_ratios_)
        return self.weight_masks_

    def gen_masks(self):
        masks = []
        for m in self.model_.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                shape = m.weight.data.size()
                masks.append(torch.zeros(shape, dtype=torch.bool))
        return masks

class ADMMPruner():
    def __init__(self, model, args, trainer, criterion, optimizer, input_shape, sparsities_maker, mask_maker, n_iterations=5, n_training_epochs=10):
        self.model_ = model
        self.args_ = args
        self.trainer_ = trainer
        self.criterion_ = criterion
        self.optimizer_ = optimizer
        self.n_iter_ = n_iterations
        self.n_training_epochs_ = n_training_epochs
        self.mask_maker_ = mask_maker
        self.sparsities_maker_ = sparsities_maker

    def projection(self, weight, wrapper):
        wrapper_cpy = copy.deepcopy(wrapper)
        wrapper_cpy.weight.data = weight
        sparsity = self.sparsities_maker_.get_sparsities()
        mask = self.mask_maker_.get_masks(sparsity)
        return prune_weight_layer(wrapper_cpy, mask)

    def compresss(self):
        logger_.info('Start AMDD pruning ...')
        # initiaze Z, U
        # Z_i^0 = W_i^0
        # U_i^0 = 0
        self.Z = []
        self.U = []
        for m in enumerate(self.model_.modules()):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                z = m.weight.data
                self.Z.append(z)
                self.U.append(torch.zeros_like(z))

        # Loss = cross_entropy +  l2 regulization + \Sum_{i=1}^N \row_i ||W_i - Z_i^k + U_i^k||^2
        # optimization iteration
        for k in range(self.n_iter_):
            logger_.info('ADMM iteration : %d', k)

            # step 1: optimize W with AdamOptimizer
            for epoch in trange(1, self.n_training_epochs_+1):
                self.trainer_(self.model_, optimizer=self.optimizer_, criterion=self.criterion_, epoch=epoch)

            # step 2: update Z, U
            # Z_i^{k+1} = projection(W_i^{k+1} + U_i^k)
            # U_i^{k+1} = U^k + W_i^{k+1} - Z_i^{k+1}
            node_idx = 0
            for m in enumerate(self.model_.modules()):
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    z = m.weight.data + self.U[node_idx]
                    self.Z[node_idx] = self.projection(z, m)
                    torch.cuda.empty_cache()
                    self.U[node_idx] = self.U[node_idx] + m.weight.data - self.Z[node_idx]
        return self.model_

class Prune_Op():
    def __init__(self, model, train_loader, criterion, input_shape, args, evaluate_function, admm_params=None, evaluate=False):
        global logger_
        logger_ = set_logger(args.debug)
        self.width = math.prod(args.group)
        self.train_loader = train_loader
        self.criterion = criterion
        self.args = args
        self.input_shape = input_shape
        self.names, self.output_shapes = activation_shapes(model, input_shape)
        self.mask_maker = MaskMaker(model, args, input_shape)
        if model.weights_pruned == None:
            model.weights_pruned = self.mask_maker.gen_masks()
        self.sparsities_maker = SimulatedAnnealing(model, start_temp=100, stop_temp=20, cool_down_rate=0.9, perturbation_magnitude=0.35, target_sparsity=0.35, args=args, evaluate_function=evaluate_function, input_shape=self.input_shape, output_shapes=self.output_shapes, mask_maker=self.mask_maker)
        if not evaluate:
            if args.admm and admm_params != None:
                pruner = ADMMPruner(model=model, args=args, trainer=admm_params['train_function'], criterion=admm_params['criterion'], optimizer=admm_params['optimizer'], input_shape=self.input_shape, mask_maker=self.mask_maker, sparsities_maker=self.sparsities_maker)
                model = pruner.compresss()
            model.weights_pruned = self.mask_maker.get_masks(self.sparsities_maker.get_sparsities())

        self.weights_pruned = model.weights_pruned
        self.model = model
        self.print_info()
        self.prune_weight()
        return

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
