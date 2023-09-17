import cv2
import numpy as np
import tensorflow as tf 
from collections import deque
import time
from multiprocessing import Process, Queue

class PoseEstimator():
    def __init__(self) -> None:
        idxs = []
        for i in range(17):
            idxs.append(3*i)
            idxs.append(3*i+1)
        self.cam_queue = deque()
        self.vid_queue = deque()
        self.quanta = [1,1]
        self.results = ({},{}) # 0 for cam, 1 for yt
        # set up so that inference in background
        self.inputs = Queue()
        self.outputs = Queue()
        self.proc = Process(target=self.polling,args=(self.inputs, self.outputs))
        self.proc.start()
    
    def pass_to_proc(self, image, key, mode):
        """
        Use this to queue inputs to the model

        Params:
            image: np array of shape HxWx3 and channels BGR (default for opencv)
            key: frame key
            mode: 0 for webcam, 1 for youtube
        """
        self.inputs.put((image, key, mode))
    
    def query(self, key, mode):
        """
        Use this to check for model results of a given key and mode
        Side effect is to empty the output queue from the model
        
        Params:
            key: frame key
            mode: 0 for webcam, 1 for youtube
        Returns:
            17x3 tensor (y,x,score) normalized to [0,1] (of padded image?)
        """
        while True:
            try:
                msg = self.outputs.get_nowait()
            except Exception:
                break
            result, k, m = msg
            self.results[m][k] = result

        if key in self.results[mode]:
            return self.results[mode][key]
        else:
            return None # Deferred??

    def kill(self, key, mode):
        """
        Call this when done with the model to allow process to terminate
        """
        self.inputs.put("Done")
        while self.query(key, mode) is None:
            continue
        self.proc.join()

    def load_model(self, ckpt=None):
        self.movenet = tf.lite.Interpreter(model_path='model.tflite')
        self.movenet.allocate_tensors()

    def add_to_queue(self, image, key, mode):
        """
        image should be tensor of shape HxWx3
        mode: 0 for webcam, 1 for video
        """
        if mode == 0:
            self.cam_queue.append((image, key))
        else:
            self.vid_queue.append((image, key))

    def polling(self, inputs, outputs):
        self.load_model()
        self.inp_details = self.movenet.get_input_details()[0]['index']
        self.out_details = self.movenet.get_output_details()[0]['index']
        while True:
            msg = inputs.get()
            if msg:
                if msg == "Done":
                    break
                image, key, mode = msg
                self.add_to_queue(image, key, mode)
            # do inference if possible
            for i, queue in enumerate([self.cam_queue, self.vid_queue]):
                if self.quanta[i] > 0 and len(queue) > 0:
                    self.quanta[i] -= 1
                    img, k = queue.popleft()
                    result = self.predict_image(img)
                    print('hi')
                    outputs.put((result, k, i))
            self.quanta[0] = min(self.quanta[0] + 1, 6)
            self.quanta[1] = min(self.quanta[1] + 1, 6)

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
    """
    image = cv2.cvtColor(cv2.imread("pose_test.png"), cv2.COLOR_BGR2RGB)
    print(image.shape)
    model.pass_to_proc(image, 1, 0)
    print(model.query(1,0))
    print(model.query(0,1))
    time.sleep(3)
    model.pass_to_proc(image, 2, 1)
    print(model.query(1,0))
    time.sleep(0.3)
    print(model.query(2,1))
    print("------------")
    num_iter=100
    start = time.time()
    for i in range(num_iter):
        model.pass_to_proc(image, i, 0)
    print("Finished adding")
    out = model.query(num_iter-1, 0)
    while out is None:
        time.sleep(0.1)
        out = model.query(num_iter-1, 0)
    print(out)
    print(f"Time elapsed for {num_iter} iterations: {time.time() - start}") 
    """
    import glob
    # files = glob.glob("pose_samples/*")
    # for file in files:
    #     image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    #     model.pass_to_proc(image, file, 0)
    # results = []
    # for file in files:
    #     result = model.query(file, 0)
    #     while result is None:
    #         time.sleep(0.1)
    #         result = model.query(file, 0)
    #     results.append({
    #         "file": file,
    #         "output": result.tolist(),
    #     })
    # import json
    # with open('pose_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)

    model.kill()

