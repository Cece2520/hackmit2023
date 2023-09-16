import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from collections import deque
import time
from multiprocessing import Process, Queue

class PoseEstimator():
    def __init__(self) -> None:
        self.load_model()
        idxs = []
        for i in range(17):
            idxs.append(3*i)
            idxs.append(3*i+1)
        self.cam_queue = deque()
        self.vid_queue = deque()
        self.cam_quanta = 0
        self.results = ({},{}) # 0 for cam, 1 for yt
        self.inp_details = self.movenet.get_input_details()[0]['index']
        self.out_details = self.movenet.get_output_details()[0]['index']

        """
        # set up so that inference in background
        self.inputs = Queue()
        self.outputs = Queue()
        self.proc = Process(target=self.polling,args=(self.inputs, self.outputs))
        self.proc.start()
        """

    def load_model(self, ckpt=None):
        self.movenet = tf.lite.Interpreter(model_path='model.tflite')
        self.movenet.allocate_tensors()
    
    def kill(self):
        self.proc.join()

    def add_to_queue(self, image, key, mode):
        """
        image should be tensor of shape HxWx3 or 1xHxWx3
        mode: 0 for webcam, 1 for video
        """
        if mode == 0:
            self.cam_queue.append((image, key))
        else:
            self.vid_queue.append((image, key))

    def polling(self, inputs, outputs):
        pass 

    def query(self, key, mode):
        if key in self.results[mode]:
            return self.results[mode][key]
        else:
            return None # Deferred??

    def predict_image(self, image):
        """
        image should be HxWx3
        output size 17x3 tensor (y,x,score) normalized to [0,1]
        """
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(tf.image.resize_with_pad(image, 192, 192), tf.uint8)
        self.movenet.set_tensor(self.inp_details, image.numpy())
        self.movenet.invoke()
        keypoints = self.movenet.get_tensor(self.out_details).squeeze()
        return keypoints
        

if __name__=="__main__":
    model = PoseEstimator()
    image = cv2.cvtColor(cv2.imread("pose_test.png"), cv2.COLOR_BGR2RGB)
    print(image.shape)
    output = model.predict_image(image)
    print(output)

    image2 = image[:,::-1,:]
    print(model.predict_image(image2))
    num_iter=100
    start = time.time()
    for i in range(num_iter):
        model.predict_image(image)
    print(f"Time elapsed for {num_iter} iterations: {time.time() - start}") 
    #model.kill()

