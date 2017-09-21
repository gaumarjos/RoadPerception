import cv2
import numpy as np
import time
import glob
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip

from includes.car_classification import *
from includes.car_detection import *
from includes.lanes_camera_calibration import *
from includes.lanes_detection import *


class RoadPerception():
    def __init__(self,
                 camera_intrinsic_calibration_filename,
                 camera_intrinsic_calibration_folder):
                 
        self.camera_intrinsic_calibration_filename = camera_intrinsic_calibration_filename
        self.camera_intrinsic_calibration_folder = camera_intrinsic_calibration_folder
        
        # Camera intrinsic calibration parameters (distortion)
        try:
            cal = pickle.load(open(self.camera_intrinsic_calibration_filename, "rb"))
            self.mtx = cal["mtx"]
            self.dist = cal["dist"]
        except:
            print("WARNING: Couldn't load camera calibration file. Generate one using generate_intrinsic_calibration")
            self.mtx = None
            self.dist = None
            
        # Camera extrinsic calibration parameters (mounting on the car)
        h = 1218
        w = 1920
        side = 300
        self.extrinsic_src = np.float32([[952, 700],
                                         [w-848, 700],
                                         [300, 1050],
                                         [w-300, 1050]])
        self.extrinsic_dst = np.float32([[side, 0],
                                         [w-side, 0],
                                         [side, h],
                                         [w-side, h]])
                                         
        # Lane detector object
        self.lane_detector = LaneDetection(mtx=self.mtx,
                                           dist=self.dist,
                                           extrinsic_src=self.extrinsic_src,
                                           extrinsic_dst=self.extrinsic_dst,
                                           video_output=True,
                                           video_output_with_camera_background=True)
        
        # Car detector object
        self.car_detector = YOLOVehicleDetector(use_tracking=False,
                                                video_output=False,
                                                video_output_with_camera_background=False)
                                         
                                             
    def generate_intrinsic_calibration(self):
        self.mtx, self.dist = camera_calibration(img_size=[1920, 1218],
                                                 calibration_filenames=self.camera_intrinsic_calibration_folder + '*.jpg',
                                                 verbose=False)
        return
        
    
    def video_processing_pipeline(self, frame):
        # Apply intrisic camera calibration (undistort)
        undistorted_frame = undistort_image(frame, self.mtx, self.dist)
              
        # Process frame
        lane_result = self.lane_detector.search_in_image(undistorted_frame)
        car_result = self.car_detector.search_in_image(undistorted_frame)
                                              
        # Draw bounding boxes (graphically merge the two results)
        result = draw_boxes(lane_result["img"], car_result["bboxes"], color=(0,0,255), thick=6)
        
        return result
    
    
    def process_video(self,
                      video_filename,
                      mode='entirevideo'):
        
        # Default output name
        video_output_filename = video_filename[:-4] + '_output.mp4'

        # Open video
        clip = VideoFileClip(video_filename)
        
        # Processing modes
        if mode == 'entirevideo':
            output_clip = clip.fl_image(self.video_processing_pipeline)#.subclip(0,10)
        
        elif mode == 'framebyframe':
            processed_frames = []
            fcount = 1
            t0 = time.time()
            for frame in clip.iter_frames():
                
                if fcount > 250:
                    break
                
                processed_frames.append(self.video_processing_pipeline(frame))
                
                # Frame counter
                if fcount % 25 == 0:
                    print("Frame #{}".format(fcount))
                fcount += 1
            
            # Time stats
            time_per_frame = (time.time() - t0) / (fcount - 1)
            print("Average time per frame: {:6.4f}s (ratio to real time: {:4.2f})" \
                  .format(time_per_frame, time_per_frame / (1./25.)))
            
            # Generate output sequence
            output_clip = ImageSequenceClip(processed_frames, fps=25)
        
        # Generate video
        output_clip.write_videofile(video_output_filename, fps=25, audio=False)
        
        return


"""
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
"""
