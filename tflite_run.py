import sys

import numpy as np
import tensorflow as tf

from utils import load_data_cifar10

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="squeezenet_cifar10_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
labels, images = load_data_cifar10(sys.argv[1], limit=None)
correct = 0
for label, image in zip(labels, images):
    input_data = np.array(image, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if np.argmax(output_data) == label:
        correct += 1

total = len(labels)
print(f'correct={correct} total={total} rate={correct/total}')
