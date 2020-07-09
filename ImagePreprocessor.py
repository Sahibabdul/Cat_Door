import numpy as np
import cv2

class ImagePreprocessor:

    def __init__ (dir, pretrained_path):
        self.dir = dir
        self.detector = cv2.CascadeClassifier(pretrained_path)

    
    # loads image from path, rbg channels
    def load_image(path):
    # load image from path
    return cv2.imread(path, cv2.COLOR_BGR2RGB)

    # loads and resizes an image to fir IMAGE_SHAPE
    def process_image(self, folder, image):
        height, width, channel
        path = path.join()s = IMAGE_SHAPE
        
    # load and resize image
        image = load_image(path)
        image = cv2.reself, size(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return image                                        