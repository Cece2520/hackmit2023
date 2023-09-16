import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow_hub as hub
from collections import deque

class PoseEstimator():
    def __init__(self) -> None:
        self.load_model()
        self.slice = []
        for i in range(17):
            self.slice.append(3*i)
            self.slice.append(3*i+1)
        self.cam_queue = deque()
        self.vid_queue = deque()
        self.cam_quanta = 0
    
    def load_model(self, ckpt=None):
        if ckpt is None:
            ckpt = "https://tfhub.dev/google/movenet/multipose/lightning/1"
        model = hub.load(ckpt)
        self.movenet = model.signatures["serving_default"]
        
    def add_to_queue(self, image, mode):
        """
        image should be tensor of shape HxWx3 or 1xHxWx3
        mode: 0 for webcam, 1 for video
        """
        if mode == 0:
            self.cam_queue.append(image)
        else:
            self.vid_queue.append(image)

    def poll(self):
        pass

    def query(self, key):
        pass

    def predict_image(self, image):
        """
        image should be HxWx3 or 1xHxWx3
        output size 34 tensor 17 (y,x) normalized to [0,1]
        """
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
        keypoints = self.movenet(image)['output_0']  # 1x6x56
        return keypoints[0,0,self.slice]
        

if __name__=="__main__":
    pass
