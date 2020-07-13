# Import System Bullshit
import sys
import os.path as path
import cv2

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

MODEL_NAME = "cnnV2"

ipp = ImagePreprocessor("images", "preprocessed_images", path.join("pretrained_models", "haarcascade_frontalcatface_extended.xml"))
cnn = CNN(MODEL_NAME)

#img = ipp.load_image(path.join("images", "cat1", "photo_2020-06-15_20-23-59.jpg"))
#img = ipp.load_image(path.join("images", "cat2", "photo_2020-06-17_17-57-11.jpg"))
#img = ipp.load_image(path.join("images", "cat2", "photo_2020-06-17_17-57-27.jpg"))

#img = ipp.load_image(path.join("images", "cat1", "photo_2020-06-15_20-23-37.jpg"))
#img = ipp.load_image(path.join("images", "cat1", "photo_2020-06-15_20-23-31.jpg"))
#ipp.detect_face(img)


if "-train" in sys.argv:
    print("TRAINING")
    print(cnn.model.summary())
    
    X, Y = ipp.load_train_data()
    cnn.train(X, Y)
    
elif "-predict" in sys.argv:
    print("PREDICT")

    X, Y = ipp.load_train_data()
    cnn.predict(X, Y)

elif "-preprocess" in sys.argv:
    print("PREPROCESS")

    ipp.preprocess_all_images()
    

        
    
else:
    print("invalid argument")
    print("python main.py -[train|predict|preprocess]")