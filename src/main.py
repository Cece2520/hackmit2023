from tkinter import *
from PIL import ImageTk, Image
import cv2
import tensorflow as tf
import time
import numpy as np
import yt_dlp
from video_data import *

def geturl():
    video_url = ent.get()
    timestamp.set(0)
    exacttime.set(0)
    url, fps = get_metadata(video_url)
    videofps.set(fps)
    framenum.set(0)
    capurl.set(url)
    # print(url)  
    
def playPause():
    start = startpause['text']
    startpause['text'] = 'Pause' if start == 'Start' else 'Start'
    prevtime.set(time.time())
    # Other things 
    
root = Tk()

timestamp = IntVar(master=root)
exacttime = DoubleVar(master=root)
prevtime = DoubleVar(master=root)
videofps = DoubleVar(master=root)
framenum = IntVar(master=root)
capurl = StringVar(master=root)
totalseconds = IntVar(master=root)

link = Frame(master=root, bg='white')
ent = Entry(master=link, width=80, bg='white')
ent.pack(padx=5, pady=5)

btn = Button(master=link, text='Load', width=10, bg='white', command=geturl)
btn.pack(padx=5, pady=5)
link.pack(fill=X)

videos = Frame(master=root, bg='white')
videos.columnconfigure([0,1], minsize=640)

youtube = Frame(master=videos, bg='white')
youtube.grid(row=0, column=0, padx=50)
youtubel = Label(master=youtube, bg='white')
youtubel.pack(pady=5)

webcam = Frame(master=videos, bg='white')
webcam.grid(row=0, column=1, padx=50, pady=5)
webl = Label(master=webcam, width=640, bg='white')
webl.pack(pady=5)

videos.pack()

videoplayer = Frame(master=root, bg='white')
startpause = Button(master = videoplayer, text='Start', width=6, bg='white', command=playPause)
startpause.pack(pady=5)
slider = Scale(master = videoplayer, variable=timestamp, from_=0, to=0, orient='horizontal', bg='white')
slider.pack(fill=X, pady=5, expand=True)

videoplayer.pack(fill=X)
scorefrm = Frame(master=root, bg='white')
scorelbl = Label(master=scorefrm, text='N/A',bg='white')
scorelbl.pack(pady=5)
scorefrm.pack(fill=X)

# Capture from camera
webcap = cv2.VideoCapture(0)
videocap = [None]

# function for video streaming
def video_stream():
    _, frame = webcap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    webl.imgtk = imgtk
    webl.configure(image=imgtk)
    # youtubel.imgtk = imgtk
    # youtubel.configure(image=imgtk)
    if startpause['text'] == 'Pause':
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
            ret, frame = videocap[0].read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                youtubel.imgtk = imgtk
                youtubel.configure(image=imgtk)
            else:
                startpause['text'] == 'Start'
    videos.after(1, video_stream) 
    

video_stream()
root.mainloop()