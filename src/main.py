from tkinter import *
from customtkinter import *
from PIL import ImageTk, Image
import cv2
import tensorflow as tf
import time
import numpy as np
import yt_dlp
from video_data import *
from model import *

def geturl():
    video_url = ent.get()
    timestamp.set(0)
    exacttime.set(0)
    url, fps = get_metadata(video_url)
    print(fps)
    videofps.set(fps//3)
    framenum.set(0)
    capurl.set(url)
    
def playPause():
    start = startpause.cget('text')
    startpause.configure(text='Pause' if start == 'Start' else 'Start')
    if start == 'Start':
        time.sleep(3)
    prevtime.set(time.time())

def quittkinter():
    model.kill()
    root.destroy()
    
root = Tk()

timestamp = IntVar(master=root)
exacttime = DoubleVar(master=root)
prevtime = DoubleVar(master=root)
videofps = DoubleVar(master=root)
framenum = IntVar(master=root)
capurl = StringVar(master=root)
totalseconds = IntVar(master=root)

link = CTkFrame(master=root)
linklbl = CTkLabel(master=link, text="Insert Youtube Link Here!")
link2 = CTkFrame(master=link)
ent = CTkEntry(master=link2, width=640)
ent.pack(side=LEFT, padx=5, pady=5)

btn = CTkButton(master=link2, text='Load', width=60, command=geturl)
btn.pack(side=LEFT, padx=5, pady=5)
linklbl.pack()
link2.pack()
link.pack(fill=X)

videos = CTkFrame(master=root)
videos.columnconfigure([0,1], minsize=640)

youtube = CTkFrame(master=videos)
youtube.grid(row=0, column=0, padx=50)
youtubel = CTkLabel(master=youtube, text='')
youtubel.pack(pady=5)

webcam = CTkFrame(master=videos)
webcam.grid(row=0, column=1, padx=50, pady=5)
webl = CTkLabel(master=webcam, width=640, text='')
webl.pack(pady=5)

videos.pack(fill=X)

videoplayer = CTkFrame(master=root)
startpause = CTkButton(master = videoplayer, text='Start', width=60, command=playPause)
startpause.pack(pady=5)
# slider = Scale(master = videoplayer, variable=timestamp, from_=0, to=0, orient='horizontal')
# slider.pack(fill=X, pady=5, expand=True)
videoplayer.pack(fill=X)

scorefrm = CTkFrame(master=root)
scorelbl = CTkLabel(master=scorefrm, text='N/A',font=('Calibri', 30))
scorelbl.pack(pady=5)
scorefrm.pack(fill=X)

quitfrm = CTkFrame(master=root)
quit = CTkButton(quitfrm, text='Quit', width=60, command=quittkinter)
quit.pack(pady=5)
quitfrm.pack(fill=X)

# Capture from camera
webcap = cv2.VideoCapture(0)
videocap = [None]
model = None

# function for video streaming
def video_stream():
    _, frame1 = webcap.read()
    frame1 = cv2.flip(frame1, 1)
    cv2image = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    webl.imgtk = imgtk
    webl.configure(image=imgtk)
    if startpause.cget("text") == 'Pause':
        if capurl.get():
            print('here')
            videocap[0] = cv2.VideoCapture(capurl.get())
            totalseconds.set(videocap[0].get(cv2.CAP_PROP_FRAME_COUNT) // videofps.get() + 1)
            slider['to'] = totalseconds.get()
            capurl.set('')
        # print(videocap[0])
        timer = time.time()
        newtime = exacttime.get() + timer - prevtime.get()
        exacttime.set(newtime)
        timestamp.set(int(newtime))
        prevtime.set(timer)
        if int(newtime*videofps.get()) > framenum.get():
            framenum.set(framenum.get() + 1)
            ret, frame2 = videocap[0].read()
            if ret:
                if framenum.get() % 3 == 0:
                    model.pass_to_proc(frame1, framenum.get(), 0)
                    model.pass_to_proc(frame2, framenum.get(), 1)
                    score = model.query()
                    if score is not None:
                        scorelbl.configure(text = str(score))
                cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                youtubel.imgtk = imgtk
                youtubel.configure(image=imgtk)
            else:
                startpause.configure(text='Start')
        # q = model.query(framenum.get(), 0)
    videos.after(10, video_stream) 
    
if __name__ == '__main__':
    model = PoseEstimator()
    video_stream()
    root.mainloop()
