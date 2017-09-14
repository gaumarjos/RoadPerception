from includes.wrapper import *

video_filename = 'videos/quick_test/test.mp4'

perception = RoadPerception(camera_intrinsic_calibration_filename='camera_calibration/sekonix120.p',
                            camera_intrinsic_calibration_folder='camera_calibration/sekonix120/')

perception.process_video(video_filename)
