import threading
import time
from camera_detector.camera_main import Detector2D
from lidar_detector.lidar_main import Detector3D

import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)


def detect_and_update_2D(detector):
    while True:
        # Call the detect method to update person_bboxes
        detector.run()

        # Wait for some time before the next update
        time.sleep(0.0001)

def detect_and_update_3D(detector):
    while True:
        # Call the detect method to update person_bboxes
        detector.run()

        # Wait for some time before the next update
        time.sleep(0.005)

def main():
    # Create a thread for camera detector
    detector_2d = Detector2D()
    thread_camera = threading.Thread(target=detect_and_update_2D, args=(detector_2d,))
    thread_camera.start()

    # Create a thread for camera detector
    detector_3d = Detector3D()
    thread_lidar = threading.Thread(target=detect_and_update_3D, args=(detector_3d,))
    thread_lidar.start()

    # Wait to initialize model
    time.sleep(0.2)

    # Start sensor fusion
    while True:
        # request camera result
        camera_result = detector_2d.person_bboxes
        if camera_result:
            print('camera result is:', camera_result)
            detector_2d.person_bboxes = []

        # request lidar result
        lidar_result = detector_3d.results
        if lidar_result:
            print('lidar result is:', lidar_result)
            detector_3d.results = []
        
        # Wait for some time before the next update
        time.sleep(0.005)


if __name__ == '__main__':
    main()