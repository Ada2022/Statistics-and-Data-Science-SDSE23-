import threading
import time
import cv2
import matplotlib.pyplot as plt
from camera_detector.camera_main import Detector2D
from lidar_detector.lidar_main import Detector3D
# from sensor_fusion.fusion_main import Matcher

def detect_and_update(detector):
    while True:
        # Call the detect method to update person_bboxes
        detector.run()

        # Wait for some time before the next update
        time.sleep(0.005)

def main():
    # Create a thread for camera detector
    detector_2d = Detector2D()
    thread_camera = threading.Thread(target=detect_and_update, args=(detector_2d,))
    thread_camera.start()

    # Create a thread for camera detector
    detector_3d = Detector3D()
    thread_lidar = threading.Thread(target=detect_and_update, args=(detector_3d,))
    thread_lidar.start()

    # Wait to initialize model
    time.sleep(0.3)

    # Initialize matcher
    # matcher = Matcher()

    # Start sensor fusion
    while thread_camera.is_alive() and thread_lidar.is_alive():
        # request camera result
        camera_result = detector_2d.person_bboxes

        # request lidar result
        lidar_result = detector_3d.results

        if camera_result and lidar_result:
            print('camera result is:', camera_result)
            print('lidar result is:', lidar_result)


            # sensor fusion 
            # matcher.run(camera_result, lidar_result)

        # Wait for some time before the next update
        time.sleep(0.005)


if __name__ == '__main__':
    main()
