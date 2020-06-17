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
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Lambda, Input

# Predefining Parameters
LEARNING_RATE = 0.00001
IMAGE_SHAPE = (128, 128, 3)
EPOCHS = 5

# loads image from path, rbg channels
def load_image(path):
    # load image from path
    return cv2.imread(path, cv2.COLOR_BGR2RGB)

# loads and resizes an image to fir IMAGE_SHAPE
def process_image(path):
    height, width, channels = IMAGE_SHAPE
    
    # load and resize image
    image = load_image(path)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return image

# Defining functions to call when starting the programm
    # Traning --> training the Neural Network
    # Predict --> using the created neural Network
    # Preprocess --> preprocessing images and getting everything ready
# Else invalid and exit
if len(sys.argv) < 2:
    print("\nnot enough arguments")
    print("python main.py -[train|predict|preprocess]")
    sys.exit(0)

if "-train" in sys.argv:
    print("TRAINING")
    
elif "-predict" in sys.argv:
    print("PREDICT")

elif "-preprocess" in sys.argv:
    print("PREPROCESS")

    # counter to uniquely identify images
    COUNTER = 0
    indxs = []
    
    # go trough all folder in images
    for folder in os.listdir(path.normpath("images")):
        print("processing images from: " + str(folder))
        
        # if folder doesn't exist in preprocessed yet, make it
        if not path.exists(path.join("preprocessed_images", folder)):
            os.mkdir(path.join("preprocessed_images", folder))
        
        # delete all old images
        for image in tqdm(os.listdir(path.join("preprocessed_images", folder))):
            os.remove(path.join("preprocessed_images", folder, image))

        # replace them with resized ones and every image gets and unique id from 0 to n images
        for image in tqdm(os.listdir(path.join("images", folder))):
            img_path = path.join("images", folder, image)
            processed_image = process_image(img_path)
            cv2.imwrite(path.join("preprocessed_images", folder, str(COUNTER) + ".png"), processed_image)

            COUNTER = COUNTER + 1
        # Allways append the current counter of images after done in a certain Folder
        indxs.append(COUNTER)

    print(indxs[-1])
    
    # make triplets
    triplets = []
    low = 0
    for high in indxs:
        for i in range(low, high):
            for j in range(low, high):
                if not i == j:
                    triplets.append([i, j, 1])
        
            for j in range (0, low):
                triplets.append([i, j, 0])
            
            for j in range(high, indxs[-1]):
                triplets.append([i, j, 0])
        low = high
    
    triplets = pd.DataFrame(triplets, columns=['IMAGE_A','IMAGE_B','Label'])
    print(triplets)
    triplets.to_csv("triplets.csv", index=False, header=True)

    

        
    
else:
    print("invalid argument")
    print("python main.py -[train|predict|preprocess]")