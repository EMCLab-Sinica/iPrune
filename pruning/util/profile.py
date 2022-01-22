from prettytable import PrettyTable
import math
'''
model_config: [
    {
        'input': [H,W,N],
        'filter': [K,K,N,M],
        'output': [R,C,M],
        'tile': {
            'input': [Th,Tw,Tn],
            'weight': [Tk,Tk,Tn,Tm],
        },
        'group': [H, W, N]
    },
    ...
]
'''
class Energy_model:
    def __init__(self, model_config):
        self.model_config = model_config

    def profile(self, layer_id):
        layer = self.model_config[layer_id]
        inputs = layer['input']
        filters = layer['filter']
        output = layer['output']
        input_tile = layer['tile']['input']
        weight_tile = layer['tile']['weight']
        stride = layer['stride']

        # Average, per ifm element is used kernel times
        nvm_access_per_ifm = (filters[3] / weight_tile[3]) * math.ceil(weight_tile[1]/stride)
        n_inputs = (inputs[0] * inputs[1] * inputs[2])
        # weight stationary
        nvm_access_per_weight = 1
        n_weights = filters[0] * filters[1] * filters[2] * filters[3]
        # #  (R*C * (#memory accesses per job)) * (#filter)
        vm_access_ifm = (output[0] * output[1] * (filters[0] * filters[1] * filters[2])) * filters[3]
        # min(output[0], math.ceil(weight_tile[0] / stride)) * min(output[1], math.ceil(weight_tile[1] / stride)) * weight_tile[3]
        vm_access_weight = vm_access_ifm
        # min(output[0], math.ceil(input_tile[0] / stride)) * min(output[1], math.ceil(input_tile[1] / stride))

        NVM_access = nvm_access_per_ifm * n_inputs + nvm_access_per_weight * n_weights
        VM_access = vm_access_ifm + vm_access_weight
        #(nvm_access_per_ifm * n_inputs) * (vm_access_per_ifm) + \
        #            (nvm_access_per_weight * n_weights) * (vm_access_per_weight)

        group = layer['group']
        n_group_per_filter = (filters[0] / group[0]) * (filters[1] / group[1]) * (filters[2] / group[2])
        # (#groups) * (RC)
        n_group = n_group_per_filter * (filters[3])
        n_vm_job_conv = n_group * output[0] * output[1]

        vm_read_per_ofm_channel = 2 * (n_group_per_filter - 1) * output[0] * output[1]
        vm_read_per_ofm = vm_read_per_ofm_channel * filters[3]

        vm_write_per_ofm_channel = (n_group_per_filter - 1) * output[0] * output[1]
        vm_write_per_ofm = vm_write_per_ofm_channel * output[2]

        n_vm_job_accum = vm_write_per_ofm + vm_read_per_ofm
        n_vm_job = n_vm_job_conv + n_vm_job_accum
        n_nvm_job = output[0] * output[1] * output[2]

        NVM_access += n_nvm_job

        VM_access += vm_read_per_ofm + vm_write_per_ofm + n_vm_job

        input_reuse = min(output[0], math.ceil(filters[0] / stride)) * min(output[1], math.ceil(filters[1] / stride)) * filters[3]
        filter_reuse = output[0] * output[1]
        MAC = filters[0] * filters[1] * filters[2] * output[0] * output[1] * output[2]

        return [layer_id, NVM_access, nvm_access_per_ifm * n_inputs, nvm_access_per_weight * n_weights, VM_access, vm_access_ifm, vm_access_weight, input_reuse, filter_reuse, MAC, n_nvm_job]

    def estimate(self):
        t = PrettyTable (['Layer id', 'NVM access - total', 'NVM access - input', 'NVM access - weight', 'VM access - total', 'VM access - input', 'VM access - weight', 'input data reuse', 'filter data reuse', "MAC", "n_nvm_job"])
        t.align["NVM access - total"] = "r"
        t.align["VM access - total"] = "r"
        t.align["MAC"] = "r"
        t.align["n_nvm_job"]
        vm_access = []
        nvm_access = []
        for layer_id in range(len(self.model_config)):
            res = self.profile(layer_id)
            t.add_row(res)
            vm_access.append(res[4])
            nvm_access.append(res[1])
        print(t)
        sorted_idx = sorted(range(len(vm_access)), key = lambda i : vm_access[i], reverse=True)
        print('sorted by vm access : {}'.format(sorted_idx))
        sorted_idx = sorted(range(len(nvm_access)), key = lambda i : nvm_access[i], reverse=True)
        print('sorted by nvm access: {}'.format(sorted_idx))
        nvm_energy_comsumption = 100
        energy = nvm_access * nvm_energy_comsumption + vm_access
        sorted_idx = sorted(range(len(nvm_access)), key = lambda i : energy[i], reverse=True)
        print('sorted by energy    : {}'.format(sorted_idx))

'''
model_config: [
    {
        'input': [H,W,N],
        'filter': [K,K,N,M],
        'output': [R,C,M],
        'tile': {
            'input': [Th,Tw,Tn],
            'weight': [Tk,Tk,Tn,Tm],
        },
        'group': [H, W, N]
    },
    ...
]
'''
class Job_model:
    def __init__(self, model_config):
        self.model_config = model_config

    def profile(self, layer_id):
        layer = self.model_config[layer_id]
        inputs = layer['input']
        filters = layer['filter']
        output = layer['output']
        input_tile = layer['tile']['input']
        weight_tile = layer['tile']['weight']
        group = layer['group']
        n_group_per_filter = (filters[0] / group[0]) * (filters[1] / group[1]) * (filters[2] / group[2])
        # (#groups) * (RC)
        n_group = n_group_per_filter * (filters[3])
        n_job_conv = n_group * output[0] * output[1]

        nvm_read_per_ofm_channel = 2 * (n_group_per_filter - 1) * output[0] * output[1]
        nvm_read_per_ofm = nvm_read_per_ofm_channel * filters[3]

        nvm_write_per_ofm_channel = (n_group_per_filter - 1) * output[0] * output[1]
        nvm_write_per_ofm = nvm_write_per_ofm_channel * output[2]

        n_progress_indicator = n_job_conv + nvm_write_per_ofm
        n_job_accum = nvm_write_per_ofm + nvm_read_per_ofm
        n_job = n_job_conv + n_job_accum

        NVM_access = n_progress_indicator + n_job

        return [layer_id, n_job, n_job_conv, n_job_accum, nvm_read_per_ofm_channel, nvm_read_per_ofm, n_group, n_group_per_filter, n_progress_indicator]

    def estimate(self):
        t = PrettyTable(["layer_id", "n_jobs", "n_job_conv", "n_job_accum", "nvm_read_per_ofm_channel", "nvm_read_per_ofm", "n_group", "n_group_per_filter", "n_progress_indicator"])
        t.align["layer_id"] = "r"
        t.align["n_jobs"] = "r"
        t.align["n_job_conv"] = "r"
        t.align["n_job_accum"] = "r"
        t.align["nvm_read_per_ofm_channel"] = "r"
        t.align["nvm_read_per_ofm"] = "r"
        t.align["n_group"] = "r"
        t.align["n_group_per_filter"] = "r"
        t.align["n_progress_indicator"] = "r"
        n_jobs = []
        n_job_conv = []
        for layer_id in range(len(self.model_config)):
            res = self.profile(layer_id)
            t.add_row(res)
            n_jobs.append(res[1])
            n_job_conv.append(res[2])
        print(t)
        sorted_idx = sorted(range(len(n_jobs)), key = lambda i : n_jobs[i], reverse=True)
        print('sorted by NVM access (intermittent execution model): {}'.format(sorted_idx))
        sorted_idx = sorted(range(len(n_job_conv)), key = lambda i : n_job_conv[i], reverse=True)
        print('sorted by #job                                     : {}'.format(sorted_idx))
'''
# 1D convolution - (left->right)
model = [
    {
        'input': [28,28,1],
        'filter': [5,5,1,8],
        'output': [28,28,8],
        'tile': {
            'input': [28,28,1],
            'weight': [5,5,1]
        },
        'group': [1, 5, 1]
    },
    {
        'input': [14,14,8],
        'filter': [5,5,8,16],
        'output': [14,14,16],
        'tile': {
            'input': [14,14,1],
            'weight': [5,5,1]
        },
        'group': [1, 5, 1]
    }
]
'''

LeNet_1 = [
    {
        'input': [32,32,1],
        'filter': [5,5,1,6],
        'output': [28,28,6],
        'tile': {
            'input': [32,32,1],
            'weight': [5,5,1,6]
        },
        'group': [1, 5, 1],
        'stride': 1
    },
    {
        'input': [14,14,6],
        'filter': [5,5,6,16],
        'output': [10,10,16],
        'tile': {
            'input': [14,14,8],
            'weight': [5,5,8,1]
        },
        'group': [1, 5, 1],
        'stride': 1
    },
    {
        'input': [5,5,16],
        'filter': [5,5,16,120],
        'output': [1,1,120],
        'tile': {
            'input': [5,5,16],
            'weight': [5,5,16,3]
        },
        'group': [1, 5, 1],
        'stride': 1
    },
    {
        'input': [1,1,120],
        'filter': [1,1,120,84],
        'output': [1,1,84],
        'tile': {
            'input': [1,1,120],
            'weight': [1,1,120,14]
        },
        'group': [1, 1, 5],
        'stride': 1
    },
    {
        'input': [1,1,84],
        'filter': [1,1,84,10],
        'output': [1,1,10],
        'tile': {
            'input': [1,1,84],
            'weight': [1,1,84,10]
        },
        'group': [1, 1, 5],
        'stride': 1
    }
]

LeNet_3 = [
    {
        'input': [28,28,1],
        'filter': [3,3,1,8],
        'output': [28,28,8],
        'tile': {
            'input': [8,3,1],
            'weight': [3,3,1,8]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    {
        'input': [14,14,8],
        'filter': [3,3,8,16],
        'output': [14,14,16],
        'tile': {
            'input': [8,3,8],
            'weight': [3,3,8,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    {
        'input': [4,4,64],
        'filter': [4,4,64,256],
        'output': [1,1,256],
        'tile': {
            'input': [4,4,64],
            'weight': [4,4,64,8]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    {
        'input': [1,1,256],
        'filter': [1,1,256,10],
        'output': [1,1,10],
        'tile': {
            'input': [1,1,128],
            'weight': [1,1,128,10]
        },
        'group': [1, 1, 2],
        'stride': 1
    }
]

AlexNet = [
    #0
    {
        'input': [224,224,3],
        'filter': [11,11,3,48],
        'output': [55,55,48],
        'tile': {
            'input': [11,11,3],
            'weight': [11,11,3,12]
        },
        'group': [1, 1, 2],
        'stride': 4
    },
    #1
    {
        'input': [55,55,48],
        'filter': [5,5,48,128],
        'output': [55,55,128],
        'tile': {
            'input': [8,5,4],
            'weight': [5,5,4,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #2
    {
        'input': [27,27,48],
        'filter': [3,3,48,192],
        'output': [27,27,192],
        'tile': {
            'input': [8,3,12],
            'weight': [3,3,12,12]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #3
    {
        'input': [13,13,192],
        'filter': [3,3,192,192],
        'output': [13,13,192],
        'tile': {
            'input': [8,3,12],
            'weight': [3,3,12,12]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #4
    {
        'input': [13,13,192],
        'filter': [3,3,192,128],
        'output': [13,13,128],
        'tile': {
            'input': [8,3,12],
            'weight': [3,3,12,12]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #5
    {
        'input': [13,13,128],
        'filter': [13,13,128,2048],
        'output': [1,1,2048],
        'tile': {
            'input': [13,13,4],
            'weight': [13,13,4,2]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #6
    {
        'input': [1,1,2048],
        'filter': [1,1,2048,2048],
        'output': [1,1,2048],
        'tile': {
            'input': [1,1,2048],
            'weight': [1,1,2048,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    }
]
SqueezeNet = [
    #0
    {
        'input': [32,32,3],
        'filter': [3,3,3,64],
        'output': [16,16,64],
        'tile': {
            'input': [8,3,3],
            'weight': [3,3,3,64]
        },
        'group': [1, 1, 2],
        'stride': 2
    },
    #1
    {
        'input': [7,7,64],
        'filter': [1,1,64,16],
        'output': [7,7,16],
        'tile': {
            'input': [7,3,64],
            'weight': [1,1,64,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #2
    {
        'input': [7,7,16],
        'filter': [1,1,16,64],
        'output': [7,7,64],
        'tile': {
            'input': [7,3,16],
            'weight': [1,1,16,64]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #3
    {
        'input': [7,7,16],
        'filter': [3,3,16,64],
        'output': [7,7,64],
        'tile': {
            'input': [7,3,16],
            'weight': [3,3,16,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #4
    {
        'input': [7,7,128],
        'filter': [1,1,128,16],
        'output': [7,7,16],
        'tile': {
            'input': [7,3,64],
            'weight': [1,1,64,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #5
    {
        'input': [7,7,16],
        'filter': [1,1,16,64],
        'output': [7,7,64],
        'tile': {
            'input': [7,3,16],
            'weight': [1,1,16,64]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #6
    {
        'input': [7,7,16],
        'filter': [3,3,16,64],
        'output': [7,7,64],
        'tile': {
            'input': [7,3,16],
            'weight': [3,3,16,16]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #7
    {
        'input': [7,7,128],
        'filter': [1,1,128,32],
        'output': [7,7,32],
        'tile': {
            'input': [7,3,64],
            'weight': [1,1,64,32]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #8
    {
        'input': [7,7,32],
        'filter': [1,1,32,128],
        'output': [7,7,128],
        'tile': {
            'input': [7,3,32],
            'weight': [1,1,32,128]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #9
    {
        'input': [7,7,32],
        'filter': [3,3,32,128],
        'output': [7,7,128],
        'tile': {
            'input': [7,3,32],
            'weight': [3,3,32,8]
        },
        'group': [1, 1, 2],
        'stride': 1
    },
    #10
    {
        'input': [7,7,256],
        'filter': [1,1,256,10],
        'output': [7,7,10],
        'tile': {
            'input': [7,3,128],
            'weight': [1,1,128,10]
        },
        'group': [1, 1, 2],
        'stride': 1
    }
]


energy_est = Energy_model(SqueezeNet)
energy_est.estimate()
jobs_est = Job_model(SqueezeNet)
jobs_est.estimate()











