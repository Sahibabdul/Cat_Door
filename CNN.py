# Import System Bullshit
import sys
import os
import os.path as path
import time
import pickle

# Import Math and Image Processing
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import actual Maching Learning Stuff
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


from ImagePreprocessor import IMAGE_SHAPE

# Predefining Parameters
LEARNING_RATE = 0.00001
EPOCHS = 5

class CNN():

    def __init__ (self, model_name, model_path=None):
        if model_path is None:
            model_path = model_name

        self.MODEL_NAME = model_name
        self.MODEL_PATH = path.join("models", model_path + ".h5")
        self.model = self.create_model()

    def create_model(self):
        input_A = Input(shape=IMAGE_SHAPE)
        input_B = Input(shape=IMAGE_SHAPE)

        preTrained = ResNet50V2(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE)

        for l in preTrained.layers[:-9]:
            l.trainable = False

        flatten1 = Flatten()(preTrained.layers[-2].output)
        dense1 = Dense(1024, activation='relu')(flatten1)

        modifiedPreTrained = Model(preTrained.input, dense1)

        output_A = modifiedPreTrained(input_A)
        output_B = modifiedPreTrained(input_B)

        l = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        l_out = l([output_A, output_B])

        dense2 = Dense(512, activation='relu')(l_out)
        output = Dense(1, activation='sigmoid')(dense2)

        model = Model(inputs=[input_A, input_B], outputs=output) 
        model.compile(loss=BinaryCrossentropy(), metrics=['acc'], optimizer=Adam(learning_rate=LEARNING_RATE))

        return model

    def save_model(self):
        self.model.save_weights(self.MODEL_PATH)

    def load_model(self):
        self.model.load_weights(self.MODEL_PATH)

    def train(self, X, Y):
        self.model.fit(X, Y, validation_split=0.2, epochs=EPOCHS, shuffle=True, batch_size=32)