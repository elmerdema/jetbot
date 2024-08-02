import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
import numpy as np

# Load your Keras model
model = load_model('model.h5')

# Define input specification
input_signature = tf.TensorSpec(shape=(1, 66, 200, 3), dtype=tf.float32, name='input')

#Convert model to onnx
spec = (tf.TensorSpec((1, 66, 200, 3), tf.float32, name="input"),) #change image size as per your trained dataset
output_path = model.name + ".onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)