#!/usr/bin/env python3

import cv2
import numpy as np
import os

class Processor:
    def __init__(self, debug_op) -> None:
        self.debug_op = debug_op
        self.norm_image = None

    def load_images(self, folder_path):
        images = []
        files = []
        for i in folder_path:
            for filename in os.listdir(i):
                if filename.endswith('.png'):
                    file_path = os.path.join(i, filename)
                    files.append(file_path)
        files.sort()

        for i in files:
            image = cv2.imread(i)
            images.append(image)
        
        for i in range(len(images)):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        print(f"Number of images loaded: {len(images)}\n")

        if self.debug_op:
            for i in range(len(images)):
                cv2.imshow(f"Grascale Image {i + 1}", images[i])
            
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return images
    
    def apply_clahe(self, clipLimit=2.0, tileGridSize=(8, 8)):
        if len(self.norm_image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            self.norm_image = clahe.apply(self.norm_image)
        else:
            lab = cv2.cvtColor(self.norm_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            self.norm_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def normalize(self, image, idx, use_clahe=False):
        if isinstance(image, np.ndarray):
            self.norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if use_clahe:
                self.apply_clahe()
        
        if self.debug_op:
            cv2.imshow(f"Normalized Image {idx + 1}", self.norm_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
        return self.norm_image