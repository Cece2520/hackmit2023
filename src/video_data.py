# Taken with modification from https://stackoverflow.com/questions/50876292/capture-youtube-video-for-further-processing-without-downloading-the-video 

import cv2
import numpy as np
import yt_dlp

def get_metadata(video_url):
    ydl_opts = {}

    # create youtube-dl object
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(video_url, download=False)
    formats = info_dict.get('formats',None)

    print(len(formats))

    for f in formats:
        if f.get('format_note',None) == '360p':

            fps = f.get('fps',None)
            url = f.get('url',None)
            return (url, fps)


def get_frames(url, fps):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print('video not opened')
        exit(-1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # DO THINGS

    cap.release()


my_url, my_fps = get_metadata("https://www.youtube.com/watch?v=lAhALtY-X2s")
get_frames(my_url, my_fps)