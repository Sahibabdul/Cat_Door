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
from keras import Model
from keras import backend as K
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D, Dense, Flatten, Lambda, Input
from keras.applications import ResNet50V2
from keras.applications.vgg16 import VGG16
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from ImagePreprocessor import IMAGE_SHAPE

# Predefining Parameters
LEARNING_RATE = 0.00000001
EPOCHS = 30

class CNN():

    def __init__ (self, model_name, model_path=None):
        if model_path is None:
            model_path = model_name

        self.MODEL_NAME = model_name
        self.MODEL_PATH = path.join("models", model_path + ".h5")
        self.model = self.create_model()

        if path.exists(self.MODEL_PATH):
            self.load_model()

    def create_model(self):
        input_A = Input(shape=IMAGE_SHAPE)
        input_B = Input(shape=IMAGE_SHAPE)

        preTrained = VGG16(include_top=False, weights='imagenet', input_shape=IMAGE_SHAPE)

        #for l in preTrained.layers[:-3]:
            #l.trainable = False

        flatten1 = Flatten()(preTrained.layers[-2].output)
        dense1 = Dense(512, activation='relu')(flatten1)

        modifiedPreTrained = Model(preTrained.input, dense1)

        output_A = modifiedPreTrained(input_A)
        output_B = modifiedPreTrained(input_B)

        l = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        l_out = l([output_A, output_B])

        #dense2 = Dense(512, activation='relu')(l_out)
        output = Dense(1, activation='sigmoid')(l_out)

        model = Model(inputs=[input_A, input_B], outputs=output) 
        model.compile(loss=BinaryCrossentropy(), metrics=['acc'], optimizer=Adam(learning_rate=LEARNING_RATE))

        return model

    def save_model(self):
        self.model.save_weights(self.MODEL_PATH)

    def load_model(self):
        self.model.load_weights(self.MODEL_PATH)

    def train(self, X, Y): 	
        tb_logs=TensorBoard(log_dir=f"logs/{self.MODEL_NAME}")
        es=EarlyStopping(monitor="val_acc",patience=3,mode=max)
        mc = ModelCheckpoint(self.MODEL_PATH, monitor='val_acc', mode='max', save_best_only=True, verbose=1)
        self.model.fit(X, Y, validation_split=0.2, epochs=EPOCHS, shuffle=True, batch_size=32, callbacks=[mc,tb_logs,es])

    def predict(self, X, Y_true):
        Y_pred = self.model.predict(X, verbose=1)

        fpr, tpr, tresholds = roc_curve(Y_true, Y_pred)
        area = auc(fpr, tpr)

        #fig=plt.figure(1)
        #plt.plot([0,1],[0,1], 'k--')
        #plt.plot(fpr, tpr, label=f'{self.MODEL_NAME} (area = {area:.3f})')

        #plt.xlabel('False positive rate')
        #plt.ylabel('True positive rate')
        #plt.title('ROC curve')
        #plt.legend(loc='best')

        #plt.savefig(f"{self.MODEL_NAME}_ROC.png",dpi=fig.dpi)

        predictions = pd.DataFrame(Y_pred)
        predictions.to_csv("predictions.csv")
