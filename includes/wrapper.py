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
        
        # Load camera calibration parameters
        try:
            cal = pickle.load(open(self.camera_intrinsic_calibration_filename, "rb"))
            self.mtx = cal["mtx"]
            self.dist = cal["dist"]
        except:
            print("WARNING: Couldn't load camera calibration file. Generate one using generate_intrinsic_calibration")
            self.mtx = None
            self.dist = None
    
    
    def generate_intrinsic_calibration(self):
        dummy = 'camera_calibration/sekonix120/0.jpg'
        dummy = cv2.imread(filename)
        dummy_size = (dummy.shape[1], dummy.shape[0])

        self.mtx, self.dist = camera_calibration(dummy_size,
                                                 calibration_filenames=self.camera_intrinsic_calibration_folder + '*.jpg',
                                                 verbose=False)
        return
    
    
    def process_video(self,
                      video_filename,
                      mode='video_moviepy'):
        
        # Default output name
        video_output_filename = video_filename[:-4] + '_output.mp4'

        if mode == 'video_moviepy':
            # Open video
            clip = VideoFileClip(video_filename)
            
            # Initialize lane detector object
            lane_detector = LaneDetection(mtx=self.mtx,
                                          dist=self.dist,
                                          full_output=False,
                                          undistort_only=True)
            
            # Initialize car detector object
            car_detector = YOLOVehicleDetector(use_tracking=False,
                                               video_output=True)
            
            # Results
            processed_frames = []
            
            # Process every frame
            fcount = 0
            t0 = time.time()
            for frame in clip.iter_frames():
                
                
                if fcount > 100:
                    break
                
                
                # Process frame
                lane_result = lane_detector.pipeline(frame)
                car_result = car_detector.search_in_image(frame)
                
                # Graphically merge the two results
                # tmp = cv2.addWeighted(img, 1.0, lane_output, 1.0, 0.0)
                # output = cv2.addWeighted(tmp, 1.0, car_output, 1.0, 0.0)
                tmp = cv2.add(frame, lane_result)
                result = cv2.add(tmp, car_result["img"])
                
                # Append the frame with bboxes drawn on it
                if car_detector.video_output:
                    processed_frames.append(result)
                
                # Frame counter
                if fcount % 25 == 0:
                    print("Frame #{}".format(fcount))
                fcount += 1
            
            # Time stats
            time_per_frame = (time.time() - t0) / fcount
            print("Average time per frame: {:6.4f}s (ratio to real time: {:4.2f})" \
                  .format(time_per_frame, time_per_frame / (1./25.)))
            
            # Generate output video from the sequence
            output_clip = ImageSequenceClip(processed_frames, fps=25)
            output_clip.write_videofile(video_output_filename, fps=25, audio=False)

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
        
        return
