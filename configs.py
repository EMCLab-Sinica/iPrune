from utils import (
    load_data_mnist,
    load_data_cifar10,
    load_data_google_speech,
    load_data_omniglot,
)

# intermediate_values_size should < 65536, or TI's compiler gets confused
configs = {
    'mnist': {
        # https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-8.onnx
        'onnx_model': 'data/mnist-8.onnx',
        'scale': 8,
        'input_scale': 8,
        'num_slots': 2,
        'intermediate_values_size': 26000,
        'data_loader': load_data_mnist,
        'n_all_samples': 10000,
        # multiply by 2 for Q15
        'sample_size': 2 * 28 * 28,
        'op_filters': 4,
        'first_sample_outputs': [ -1.247997, 0.624493, 8.609308, 9.392411, -13.685033, -6.018567, -23.386677, 28.214134, -6.762523, 3.924627 ],
        'fp32_accuracy': 0.9889,
    },
    'cifar10': {
        'onnx_model': 'data/squeezenet_cifar10.onnx',
        'scale': 8,
        'input_scale': 8,
        'num_slots': 3,
        'intermediate_values_size': 65000,
        'data_loader': load_data_cifar10,
        'n_all_samples': 10000,
        'sample_size': 2 * 32 * 32 * 3,
        'op_filters': 4,
        'first_sample_outputs': [ 0.000830, 0.000472, 0.000637, 0.678688, 0.000534, 0.310229, 0.001407, 0.003931, 0.002062, 0.001210 ],
        'fp32_accuracy': 0.7704,
    },
    'kws': {
        'onnx_model': 'data/KWS-DNN_S.onnx',
        'scale': 8,
        'input_scale': 120,
        'num_slots': 2,
        'intermediate_values_size': 20000,
        'data_loader': load_data_google_speech,
        'n_all_samples': 4890,
        'sample_size': 2 * 25 * 10,  # MFCC gives 25x10 tensors
        'op_filters': 4,
        'first_sample_outputs': [ 0.000000, 0.000000, 0.998193, 0.000000, 0.000000, 0.000000, 0.001807, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 ],
        # Much lower than reported on the paper due to mismatched window_size_ms/window_stride_ms (?)
        # See: https://github.com/ARM-software/ML-KWS-for-MCU/issues/44
        'fp32_accuracy': 0.6323,
    },
    'omniglot': {
        'onnx_model': 'data/maml.onnx',
        'scale': 4,
        'input_scale': 4,
        'num_slots': 2,
        'intermediate_values_size': 30000,
        'data_loader': load_data_omniglot,
        'n_all_samples': 5 * 20,  # 5-way (classes), each with 20 samples
        'sample_size': 2 * 28 * 28,
        'op_filters': 4,
        'first_sample_outputs': [ -0.230564, -0.879236, -0.910271, -0.212429, 0.534965 ],
        'fp32_accuracy': 0,
    },
}
