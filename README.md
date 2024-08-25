# IMU and Ultrawideband Sensor Fusion with Invariant Extended Kalman Filter

This repository presents an Invariant Extended Kalman Filter (InEKF) to perform sensor fusion of IMU and UWB data.  The dataset used is the UTIL dataset provided by the University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute)/ [UofT Robotics Institute](https://robotics.utoronto.ca/) and comparisons are made with the reference Error-State Kalman Filter (ESKF).

## Conda Environment Installation

The dependencies are managed via the `conda` package manager with packages from `conda-forge`.  Install the latest release of [`miniforge`](https://github.com/conda-forge/miniforge/releases/lates

On Linux or macOS run the following commands to install and activate the `conda` development environment:

    ./conda/setup-environment.sh
    conda activate imu-uwb-inekf

## Dataset Installation

To install the dataset run the following commands:

    curl -L https://github.com/utiasDSL/util-uwb-dataset/releases/download/dataset-v1.0/dataset.7z > dataset.7z
    tar xvf dataset.7z

## Running the Estimator Scripts

To run the estimators and generate plots run the following commands:

    cd scripts/estimation/
    python3 main.py -i ../../dataset/flight-dataset/survey-results/anchor_const1.npz ../../dataset/flight-dataset/csv-data/const1/const1-trial1-tdoa2.csv

There are 4 constellation sets (const1, const2, const3, and const4), with up to 7 trials each in both TDoA 2 and TDoA 3 (const#-trial#-tdoa#).

# Data Parsing Scripts for UTIL: Ultra-wideband Dataset

Detailed information and instructions: [https://utiasdsl.github.io/util-uwb-dataset/](https://utiasdsl.github.io/util-uwb-dataset/.)

The dataset paper:  [https://arxiv.org/pdf/2203.14471.pdf](https://arxiv.org/pdf/2203.14471.pdf)

## Dataset Usage

* Kailai Li, Ziyu Cao, and Uwe D. Hanebeck. "Continuous-Time Ultra-Wideband-Inertial Fusion." arXiv preprint arXiv:2301.09033,(2023). [Paper Link](https://arxiv.org/pdf/2301.09033.pdf), [Open Source Code](https://github.com/KIT-ISAS/SFUISE).

* Wenda Zhao, Abhishek Goudar, and Angela P. Schoellig. "Finding the right place: Sensor placement for uwb time difference of arrival localization in cluttered indoor environments." IEEE Robotics and Automation Letters 7, no. 3 (2022): 6075-6082. [Paper Link](https://ieeexplore.ieee.org/document/9750886).

## Credits

This dataset was the work of [Wenda Zhao](https://williamwenda.github.io/), [Abhishek Goudar](https://www.linkedin.com/in/abhishek-goudar-47b46090/), [Xinyuan Qiao](https://www.linkedin.com/in/xinyuan-sam-qiao-8b15ba17a/), and [Angela P. Schoellig](https://www.dynsyslab.org/prof-angela-schoellig/). If you use the data provided by this website in your own work, please use the following citation:
```
@INPROCEEDINGS{zhao2022uwbData,
      title={UTIL: An Ultra-wideband Time-difference-of-arrival Indoor Localization Dataset},
      author={Wenda Zhao and Abhishek Goudar and Xinyuan Qiao and Angela P. Schoellig},
      booktitle={International Journal of Robotics Research (IJRR)},
      year={2022},
      volume={},
      number={},
      pages={},
      doi={}
}
```
-----
> University of Toronto's [Dynamic Systems Lab](https://github.com/utiasDSL) / [Vector Institute](https://github.com/VectorInstitute)/ [UofT Robotics Institute](https://robotics.utoronto.ca/)
