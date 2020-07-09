import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import os
import os.path as path


IMAGE_SHAPE = (128, 128, 3)
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
        height, width, channel = IMAGE_SHAPE

        # load and resize image
        image = self.load_image(image_path)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return image    

    def load_train_data(self):
        triplets = pd.read_csv("triplets.csv")

        Input_A, Input_B, Labels = [], [], triplets['Label'].to_list()
        
        print ("\nLoading Images:")
        for _,row in tqdm(triplets.iterrows()):
            image_A = self.load_image(path.join(self.new_dir, str(row.at['IMAGE_A']) + ".png"))
            image_B = self.load_image(path.join(self.new_dir, str(row.at['IMAGE_B']) + ".png"))
            
            Input_A.append(image_A)
            Input_B.append(image_B)
        
        return [Input_A, Input_B], np.array(Labels)
            

    # -------- PREPROCESSING --------
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
                processed_image = self.process_image(img_path)
                cv2.imwrite(path.join(self.new_dir, str(COUNTER) + ".png"), processed_image)

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