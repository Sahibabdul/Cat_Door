# Import System Bullshit
import sys
import os
import os.path as path
import time
import pickle

# Import Math and Image Processing
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Import actual Maching Learning Stuff
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Input
from tensorflow.keras.applications import ResNet50V2

from ImagePreprocessor import ImagePreprocessor
from CNN import CNN

# Defining functions to call when starting the programm
    # Traning --> training the Neural Network
    # Predict --> using the created neural Network
    # Preprocess --> preprocessing images and getting everything ready
# Else invalid and exit
if len(sys.argv) < 2:
    print("\nnot enough arguments")
    print("python main.py -[train|predict|preprocess]")
    sys.exit(0)

MODEL_NAME = "cnnV1"

ipp = ImagePreprocessor("images", "preprocessed_images", path.join("pretrained_models", "haarcascade_frontalcatface.xml"))
cnn = CNN(MODEL_NAME)

if "-train" in sys.argv:
    print("TRAINING")
    print(cnn.model.summary())
    
    X, Y = ipp.load_train_data()
    cnn.train(X, Y)
    
elif "-predict" in sys.argv:
    print("PREDICT")

elif "-preprocess" in sys.argv:
    print("PREPROCESS")

    ipp.preprocess_all_images()
    

        
    
else:
    print("invalid argument")
    print("python main.py -[train|predict|preprocess]")