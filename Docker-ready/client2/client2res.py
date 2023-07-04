import os
import flwr as fl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from glob import glob
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

hyper_dimension = 180
IMAGE_SIZE = [hyper_dimension,hyper_dimension]
hyper_epochs = 10
hyper_batch_size = 8
hyper_feature_maps = 32
hyper_channels = 1
hyper_mode = 'grayscale'

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

train_gen = ImageDataGenerator(rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip = True,
                               vertical_flip = True)

val_gen = ImageDataGenerator(rescale = 1./255)

training_set = train_gen.flow_from_directory('train2',
                                          target_size = (hyper_dimension,
                                                         hyper_dimension),
                                          batch_size = hyper_batch_size,
                                          class_mode = 'categorical')

test_set = val_gen.flow_from_directory('val2',
                                      target_size = (hyper_dimension,
                                                     hyper_dimension),
                                      batch_size = hyper_batch_size,
                                      class_mode = 'categorical')

x_train, y_train = next(training_set)
x_test, y_test = next(test_set)

base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (180, 180, 3))
tf.keras.backend.clear_session()

for layer in base.layers:
    layer.trainable = False
    
model = Sequential()
model.add(base)
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

optm = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optm, 
                  metrics=['accuracy'])

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(training_set, epochs=hyper_epochs)
        return model.get_weights(), len(training_set), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="server:50051", client=CifarClient())