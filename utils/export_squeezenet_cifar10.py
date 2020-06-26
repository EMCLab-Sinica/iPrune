import os.path
import sys

import tensorflow as tf
import onnx
import keras2onnx

with open(os.path.join(sys.argv[1], 'models', 'squeeze_net.json')) as f:
    model_json = f.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(os.path.join(sys.argv[1], 'models', 'squeeze_net.h5'))

onnx_model = keras2onnx.convert_keras(model, model.name)

onnx.save(onnx_model, 'squeezenet_cifar10.onnx')

tf.keras.models.save_model(model, ".")

converter = tf.lite.TFLiteConverter.from_saved_model('.')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open('squeezenet_cifar10_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)
