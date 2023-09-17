import cv2
import numpy as np
import tensorflow as tf 
from collections import deque
import time
from multiprocessing import Process, Queue
from app.score import main_score

class PoseEstimator():
    def __init__(self) -> None:
        self.cam_queue = deque()
        self.vid_queue = deque()

        self.window_size = 3
        self.stacked_poses = [None, None]
        self.parity = 0

        self.results = []
        # 0 for cam, 1 for yt, 2 for sentinel
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
    
    def query(self):
        """
        Use this to check for model results of a given key and mode
        Side effect is to empty the output queue from the model
        
        Returns:
            scalar score of most recent window, or None if not done
        """
        while True:
            try:
                msg = self.outputs.get_nowait()
            except Exception:
                break
            result = msg
            self.results.append(result)

        if self.results:
            return self.results[-1]
        else:
            return None # Deferred??

    def kill(self):
        """
        Call this when done with the model to allow process to terminate
        """
        self.inputs.put("Done")
        while self.query() != "Done":
            continue
        self.proc.join()

    def load_model(self, ckpt=None):
        self.movenet = tf.lite.Interpreter(model_path='model.tflite')
        self.movenet.allocate_tensors()

    def polling(self, inputs, outputs):
        self.load_model()
        self.inp_details = self.movenet.get_input_details()[0]['index']
        self.out_details = self.movenet.get_output_details()[0]['index']
        while True:
            msg = inputs.get()
            if msg:
                if msg == "Done":
                    outputs.put("Done")
                    break
                image, key, mode = msg
                queue = [self.cam_queue, self.vid_queue][mode]
                result = self.predict_image(image)

                stack = self.stacked_poses[mode]
                if stack is None:
                    self.stacked_poses[mode] = result
                elif stack.shape[0]:
                    self.stacked_poses[mode] = np.concatenate((stack[1:,:,:], result), axis=0)
                    self.parity += 2 * mode - 1
                    if self.parity == 0:
                        outputs.put(main_score(self.stacked_poses[0], self.stacked_poses[1]).item())
                else:
                    self.stacked_poses[mode] = np.concatenate((stack, result), axis=0)


    def predict_image(self, image):
        """
        image should be HxWx3
        output size 1x17x3 tensor (y,x,score) normalized to [0,1]
        """
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(tf.image.resize_with_pad(image, 192, 192), tf.uint8)
        self.movenet.set_tensor(self.inp_details, image.numpy())
        self.movenet.invoke()
        keypoints = self.movenet.get_tensor(self.out_details)
        return keypoints[0,:,:,:]
        

if __name__=="__main__":
    model = PoseEstimator()
    
    image = cv2.cvtColor(cv2.imread("pose_test.png"), cv2.COLOR_BGR2RGB)
    print(image.shape)
    """
    model.pass_to_proc(image, 1, 0)
    model.pass_to_proc(image, 1, 1)
    model.pass_to_proc(image, 1, 0)
    model.pass_to_proc(image, 1, 1)
    model.pass_to_proc(image, 1, 0)
    model.pass_to_proc(image, 1, 1)
    print(model.query())
    time.sleep(3)
    print(model.query())
    model.pass_to_proc(image, 2, 0)
    model.pass_to_proc(image, 2, 1)
    time.sleep(0.3)
    print(model.query())
    print("------------")
    """
    num_iter=100
    start = time.time()
    for i in range(num_iter):
        model.pass_to_proc(image, i, 0)
        model.pass_to_proc(image, i, 1)
    print("Finished adding")
    #out = model.query(num_iter-1, 0)
    #while out is None:
    #    time.sleep(0.1)
    #    out = model.query(num_iter-1, 0)
    #print(out)
    model.kill()
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
    """

