# Project README

This project is developed and tested on Ubuntu 18.04 with Python 3.6.

## Running Instructions

### Creating conda environment

To create a Python 3.6 conda environment, please run the following command:

`conda create --name myenv python=3.6`

To activate the conda environment, run:

`conda activate myenv`

Then, install all necessary libraries by running:

`pip install -r requirements.txt`

### Running the Camera and Lidar Detectors

To run the camera and lidar detectors, navigate to the 'src' directory by running:

`cd ./src`

Then, run the main script by executing the following command:

`python main.py`

This will display the output of the camera and lidar detectors.

### Running the Sensor Fusion

To run the sensor fusion, navigate to the 'fusion_result' directory by running:

`cd ./sensor_fusion/fusion_result`

Then, run the script by executing the following command:

`python fusion_result.py`

This will display the output of the sensor fusion.

## Result Explanation

Due to the lack of timestamps in the dataset, we cannot align the processing frequency of the camera and lidar, which means that we cannot achieve real-time sensor fusion. Additionally, different CPUs may yield slightly different results. Therefore, we have included a standard processing result video based on the 8th generation Core i7 processor in the `demo` folder for your reference.
