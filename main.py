"""
Configurations that work:

flow --imgdir sample_img/ --model cfg/tiny-yolo-voc.cfg --load bin/thtrieu/tiny-yolo-voc.weights --gpu 0.8
flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/thtrieu/yolo.weights --gpu 0.8
flow --imgdir sample_img/ --model cfg/yolo-voc.cfg --load bin/thtrieu/yolo.weights --gpu 0.8
"""

import cv2
import numpy as np
import time
from moviepy.editor import VideoFileClip, ImageSequenceClip
from includes.car_detection import *


# Operational mode
#MODE = 'camera_cv2'
MODE = 'video_moviepy'
#MODE = 'video_cv2'
#MODE = 'image'
video_filename = 'videos/quick_test/test.mp4'
video_output_filename = video_filename[:-4] + '_output.mp4'


if MODE == 'video_moviepy':
    # Open video
    clip = VideoFileClip(video_filename)
    
    # Initialize processing object
    process = YOLOVehicleDetector(video_output=True)
    
    # Results
    processed_frames = []
    
    # Process every frame
    fcount = 0
    t0 = time.time()
    for frame in clip.iter_frames():
        
        # Process frame
        result = process.search_in_image(frame)
        
        # Append the frame with bboxes drawn on it
        if process.video_output:
            processed_frames.append(result["img"])
        
        # Frame counter
        if fcount % 25 == 0:
            print("Frame #{}".format(fcount))
        fcount += 1
    
    # Time stats
    time_per_frame = (time.time() - t0) / fcount
    print("Average time per frame: {:6.4f}s (ratio to real time: {:4.2f})".format(time_per_frame, time_per_frame / (1./25.)))
    
    # Generate output video from the sequence
    output_clip = ImageSequenceClip(processed_frames, fps=25)
    output_clip.write_videofile(video_output_filename, fps=25, audio=False)


if MODE == 'image':
    img = cv2.imread("./sample_img/sample_dog.jpg")
    pipeline(img)


if MODE == 'camera_cv2':
    # Open video stream
    clip = cv2.VideoCapture(0)
    # clip = cv2.VideoCapture("test.mp4")
    assert clip.isOpened(), 'Cannot capture source'
    
    # Initialize processing object
    process = ProcessingPipeline(video_output=False)
    
    # Process it
    fcount = 0
    while(clip.isOpened()):
        ret, frame = clip.read()
        process.pipeline(frame)
        
        fcount += 1

    clip.release()    
