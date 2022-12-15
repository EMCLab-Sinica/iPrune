from utils import (
    load_coco128,
    load_data_cifar10,
    load_data_google_speech_cnn,
    load_har,
)

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'pruned_cifar10': {
        'onnx_model': {
            'energy': '../pruning/onnx_models/energy/SqueezeNet.onnx',
            'intermittent': '../pruning/onnx_models/intermittent/SqueezeNet.onnx'
        },
        'scale': 2,
        'input_scale': 4,
        'num_slots': 3,
        'intermediate_values_size': 65000,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': [3, 32, 32],
        'op_filters': 4,
        'first_sample_outputs': [ 4.895500, 4.331344, 4.631835, 11.602396, 4.454658, 10.819544, 5.423588, 6.451203, 5.806091, 5.272837 ],
        'fp32_accuracy': 0.7704,
    },
    'pruned_kws_cnn': {
        'onnx_model': {
            'energy': '../pruning/onnx_models/energy/KWS_CNN_S.onnx',
            'intermittent': '../pruning/onnx_models/intermittent/KWS_CNN_S.onnx',
        },
        'scale': 1.6,
        'input_scale': 120,
        'num_slots': 2,
        'intermediate_values_size': 65535,
        'data_loader': load_data_google_speech_cnn,
        'n_all_samples': 4890,
        'sample_size': [1, 49, 10],  # MFCC gives 25x10 tensors
        'op_filters': 4,
        'first_sample_outputs': [ -29.228327, 5.429047, 22.146973, 3.142066, -10.448060, -9.513299, 15.832925, -4.655487, -14.588447, -1.577156, -5.864228, -6.609077 ],
        'fp32_accuracy': 0.7983,
    },
    'pruned_har': {
        'onnx_model': {
            'energy': '../pruning/onnx_models/energy/HAR.onnx',
            'intermittent': '../pruning/onnx_models/intermittent/HAR.onnx',
        },
        'scale': 2,
        'input_scale': 16,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_har,
        'n_all_samples': 2947,
        'sample_size': [9, 1, 128],
        'op_filters': 2,
        'first_sample_outputs': [ -6.194588, 2.2284777, -13.659239, -1.4972568, 13.473643, -10.446839 ],
        'fp32_accuracy': 0.9121,
    },
    'pruned_yolov5n': {
        'onnx_model': {
            'energy': '../pruning/onnx_models/energy/yolov5n.onnx',
            'intermittent': '../pruning/onnx_models/intermittent/yolov5n.onnx',
        },
        'scale': 4,
        'input_scale': 16,
        'num_slots': 6,
        'intermediate_values_size': 264000,
        'data_loader': load_coco128,
        'n_all_samples': 1,
        'sample_size': [3, 128, 128],
        'op_filters': 2,
        'first_sample_outputs': [],
        'fp32_accuracy': 0,
    },
}

