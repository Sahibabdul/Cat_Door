import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import imutils

import os
import os.path as path


IMAGE_SHAPE = (32, 32, 3)
SCALE = 1.02
NEIGHBORS = 1


class ImagePreprocessor():

    def __init__ (self, dir, new_dir, pretrained_path):
        self.dir = dir
        self.new_dir = new_dir
        self.detector = cv2.CascadeClassifier(pretrained_path)
        self.IMAGE_STORE = {}

    # ---------- DIRECTORY STUFF ----------
    def get_folders(self):
        return os.listdir(path.normpath(self.dir))
    
    def get_image_path(self, folder, image):
        return path.join(self.dir, folder, image)


    # ---------- IMAGE LOADING ----------

    def show_image(self, image):
        cv2.imshow('image',image)
        cv2.waitKey(0)

    def resize_image(self, image):
        height, width, channel = IMAGE_SHAPE
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # loads image from path, rbg channels
    def load_image(self, image_path):
        # load image from path
        if image_path in self.IMAGE_STORE:
            return self.IMAGE_STORE[image_path]
        else:
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            self.IMAGE_STORE[image_path] = image
            return image

    # loads and resizes an image to fir IMAGE_SHAPE
    def process_image(self, image_path):
        # load and resize image
        image = self.load_image(image_path)
        image = self.resize_image(image)
        return image    

    def load_train_data(self):
        triplets = pd.read_csv("triplets.csv")

        Input_A, Input_B, Labels = [], [], triplets['Label'].to_list()
        
        print ("\nLoading Images:")
        for _,row in tqdm(triplets.iterrows()):
            image_A = self.load_image(path.join(self.new_dir, str(row.at['IMAGE_A']) + ".png"))/256
            image_B = self.load_image(path.join(self.new_dir, str(row.at['IMAGE_B']) + ".png"))/256
            
            Input_A.append(image_A)
            Input_B.append(image_B)
        
        return [Input_A, Input_B], np.array(Labels)
            

    # -------- PREPROCESSING --------
    def detect_face(self, image):
        for angle in range(0, 60, 10):
            img = imutils.rotate_bound(image, angle - 30)
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            faces = self.detector.detectMultiScale(gray_scale, SCALE, NEIGHBORS)
            for (x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            
            self.show_image(img)

    def extract_face(self, image):
        best_im, best_w = image, 0

        for angle in range(0, 60, 10):
            img = imutils.rotate_bound(image, angle - 30)
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.detector.detectMultiScale(gray_scale, SCALE, NEIGHBORS)
            for (x,y,w,h) in faces:
                if w > best_w:
                    best_im, best_w = img[y:y+h, x:x+w], w

        return best_im


    def preprocess_all_images(self):
        # counter to uniquely identify images
        COUNTER = 0
        indxs = []

        # delete all old images
        for image in tqdm(os.listdir(self.new_dir)):
            os.remove(path.join(self.new_dir, image))
        
        # go trough all folder in images
        for folder in self.get_folders():
            print("processing images from: " + str(folder))
            
            # replace them with resized ones, every image gets an unique id (uids go from 0 to n)
            for image in tqdm(os.listdir(path.join(self.dir, folder))):
                img_path = self.get_image_path(folder, image)
                new_image = self.load_image(img_path)
                new_image = self.extract_face(new_image)
                new_image = self.resize_image(new_image)
                cv2.imwrite(path.join(self.new_dir, str(COUNTER) + ".png"), new_image)

                os.rename(img_path, path.join(self.dir, folder, "orig_" + str(COUNTER) + ".png"))

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
                        for i in range(10):
                            triplets.append([i, j, 1])
                        
            
                for j in range (0, low):
                    triplets.append([i, j, 0])
                
                for j in range(high, indxs[-1]):
                    triplets.append([i, j, 0])
            low = high
        
        triplets = pd.DataFrame(triplets, columns=['IMAGE_A','IMAGE_B','Label'])
        print(triplets)
        triplets.to_csv("triplets.csv", index=False, header=True)

    
