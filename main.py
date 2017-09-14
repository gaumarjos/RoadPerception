"""
Configurations that work:

flow --imgdir sample_img/ --model cfg/tiny-yolo-voc.cfg --load bin/thtrieu/tiny-yolo-voc.weights --gpu 0.8
flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/thtrieu/yolo.weights --gpu 0.8
flow --imgdir sample_img/ --model cfg/yolo-voc.cfg --load bin/thtrieu/yolo.weights --gpu 0.8
"""

from darkflow.net.build import TFNet
from darkflow.cython_utils import cy_yolo_findboxes
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import time


# Operational mode
#MODE = 'camera_cv2'
MODE = 'video_moviepy'
#MODE = 'video_cv2'
#MODE = 'image'
video_filename = 'videos/quick_test/test.mp4'
video_output_filename = video_filename[:-4] + '_output.mp4'


"""
FUNCTIONS TAKEN FROM THE CARND PROJECT
"""
# Draw bounding boxes
def draw_boxes(img, bboxes, color=(0,0,255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    

# Master pipeline function, containing all operations performed on the frames
class ProcessingPipeline():
    def __init__(self,
                 video_output):
        
        self.video_output = video_output
        
        
    def pipeline(self, img):
    
        # Default image output
        if self.video_output:
            img_out = np.copy(img)
    
        if img is not None:
            # Detect objects in frame
            detections = tfnet.return_predict(img)
            
            # Go through each detection
            bboxes = []
            if len(detections) > 0:
                for detection in detections:
                
                    # Check if the object is an interesting one
                    if detection ["label"] == "person" or \
                       detection ["label"] == "bicycle" or \
                       detection ["label"] == "car" or \
                       detection ["label"] == "motorbike" or \
                       detection ["label"] == "bus" or \
                       detection ["label"] == "train" or \
                       detection ["label"] == "truck":

                        bbox = ( (detection["topleft"]["x"], detection["topleft"]["y"]), 
                                 (detection["bottomright"]["x"], detection["bottomright"]["y"]) )
                        bboxes.append(bbox)
                
                if self.video_output:
                    img_out = draw_boxes(img, bboxes, color=(0,0,255), thick=6)
        
        # Output
        if self.video_output:
            result = {"img": img_out,
                      "bboxes": bboxes}
        else:
            result = {"img": None,
                      "bboxes": bboxes}
        
        return result
       

# Load darkflow 
options = {"model": "cfg/yolo.cfg",
           "load":  "bin/thtrieu/yolo.weights",
           "threshold": 0.3,
           "gpu": 0.9}
tfnet = TFNet(options)


if MODE == 'video_moviepy':
    # Open video
    clip = VideoFileClip(video_filename)
    processed_frames = []
    
    # Initialize processing object
    process = ProcessingPipeline(video_output=True)
    
    # Process every frame
    fcount = 0
    t0 = time.time()
    for frame in clip.iter_frames():
        if fcount > 50:
            break    
        
        # Process frame
        result = process.pipeline(frame)
        
        # Append the frame with bboxes drawn on it
        if process.video_output:
            processed_frames.append(result["img"])
            
        # print(result["bboxes"])
        
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
