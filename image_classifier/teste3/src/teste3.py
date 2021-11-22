# Imports and check that we are using TF2.x
import numpy as np
import os

from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# assert tf.__version__.startswith('2')
# tf.get_logger().setLevel('ERROR')

data = DataLoader.from_folder("C:\dev\workspaceMateus\python_IA\image_classifier\\teste3\\flower_photos")

train_data, test_data = data.split(0.9)

model = image_classifier.create(train_data)

loss, accuracy = model.evaluate(test_data)

print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

# realizando o treinamento e exportando o modelo treinado
model.export(export_dir='/mm_flowers')

response = model.evaluate("C:\dev\workspaceMateus\python_IA\image_classifier\\teste3\picture_run\\flower")

print(f"response: {response}")