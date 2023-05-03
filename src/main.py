import threading
import time
import numpy as np
from camera_detector.camera_main import Detector2D
from lidar_detector.lidar_main import Detector3D
from sensor_fusion.fusion_main import Matcher

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
    camera_demo_flag = 1
    camera_demo_result = []
    thread_camera = threading.Thread(target=detect_and_update_2D, args=(detector_2d,))
    thread_camera.start()

    # Create a thread for camera detector
    detector_3d = Detector3D()
    lidar_demo_flag = True
    lidar_demo_result = []
    thread_lidar = threading.Thread(target=detect_and_update_3D, args=(detector_3d,))
    thread_lidar.start()

    # Wait to initialize model
    time.sleep(0.2)

    # Start sensor fusion
    while True:
        # request camera result
        camera_result = detector_2d.person_bboxes
        if camera_demo_flag == 4 and camera_result:
            camera_demo_result = camera_result
            camera_demo_flag = 10e9
        if camera_result:
            print('camera result is:', camera_result)
            detector_2d.person_bboxes = []
            camera_demo_flag += 1

        # request lidar result
        lidar_result = detector_3d.results
        if lidar_demo_flag and lidar_result:
            lidar_demo_result = lidar_result
            lidar_demo_flag = False
        if lidar_result:
            print('lidar result is:', lidar_result)
            detector_3d.results = []

        while lidar_demo_result and camera_demo_result:
            lidar_results_transform = np.array([[ 9.99635086e-01,  2.69526266e-02, -1.80319433e-03, -6.48575600e-02],
                                            [-2.70109374e-02,  9.98133934e-01, -5.47636676e-02, -5.68320100e-02],
                                            [ 3.23804765e-04,  5.47923895e-02,  9.98497716e-01, -1.04481150e-01],
                                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

            camera_intrinsic_parameters = np.array([[ 611.524169921875,  0.00000000e+00, 321.3505554199219, 0.00000000e+00],
                                            [0.00000000e+00,  609.79541015625, 249.3844451904297, 0.00000000e+00],
                                            [0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00]])
            # sensor fusion 
            # matcher = Matcher(camera_demo_result, lidar_demo_result, lidar_results_transform, camera_intrinsic_parameters)
            # matcher.run()

            # run once as demo
            camera_demo_result, lidar_demo_result = None, None

        # Wait for some time before the next update
        time.sleep(0.005)


if __name__ == '__main__':
    main()
