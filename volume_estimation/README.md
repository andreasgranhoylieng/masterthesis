# Syringe Volume Estimation System

This component calculates liquid volume in syringes using keypoint detection of plunger position.

## Key Components

- `train.ipynb`: Training notebook for YOLO pose model
- `webcam_inference_pose.ipynb`: Real-time webcam volume estimation
- `pose_video_visualizer.ipynb`: Video processing with volume calculations
- `SyringeVolumeCalculator`: Core volume calculation class

## Directory Structure
```
volume_estimation/
├── train.ipynb                 # Model training
├── webcam_inference_pose.ipynb # Live webcam inference
├── pose_video_visualizer.ipynb # Video file processing
└── pose_visualizer.ipynb       # Dataset visualization
```

## Usage

1. Train model:
```bash
jupyter notebook train.ipynb
```
2. Run inference:
- For webcam:
```bash
jupyter notebook webcam_inference_pose.ipynb
```
- For video files:
```bash
jupyter notebook pose_video_visualizer.ipynb
```

## Key Features
- Real-time volume calculations using cylinder volume formula
- Median filtering for stable measurements
- Visual debugging of keypoints
- CSV export of volume timeseries

## Dependencies
- Ultralytics YOLO11
- OpenCV
- NumPy
- Pandas
