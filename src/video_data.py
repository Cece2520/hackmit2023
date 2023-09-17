# Taken with modification from https://stackoverflow.com/questions/50876292/capture-youtube-video-for-further-processing-without-downloading-the-video 

import cv2
import numpy as np
import yt_dlp
import asyncio

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
            print(url)
            return (url, fps)


def get_frames(url, fps):
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print('video not opened')
        exit(-1)
    key = -1

    while True:
        key += 1
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(30) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_frame(cap, time_in_millisec):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_millisec)
    ret, frame = cap.read()
    return frame


my_url, my_fps = get_metadata("https://youtube.com/shorts/47BLMhB6fAw?feature=shared")
get_frames(my_url, my_fps)