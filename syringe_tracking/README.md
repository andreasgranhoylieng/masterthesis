# Syringe Tracking System

This component provides object detection and tracking for multiple syringes in video streams.

## Key Components

- `train.ipynb`: YOLO object detection model training
- `webcam_inference_od.ipynb`: Real-time webcam tracking
- `video_inference_od.ipynb`: Video file tracking
- `bytetrack_od.yaml`: ByteTrack configuration

## Directory Structure
```
syringe_tracking/
├── train.ipynb            # Detection model training
├── webcam_inference_od.ipynb # Live tracking
├── video_inference_od.ipynb  # Video tracking
└── bytetrack_od.yaml      # Tracker configuration
```

## Usage

1. Train detection model:
```bash
jupyter notebook train.ipynb
```
2. Run tracking:
- For webcam:
```bash
jupyter notebook webcam_inference_od.ipynb
```
- For video files:
```bash
jupyter notebook video_inference_od.ipynb
```

## Features
- ByteTrack multi-object tracking
- 4K resolution support
- Bounding box interpolation
- Track history visualization
- Multi-GPU training support

## Dependencies
- Ultralytics YOLO11
- ByteTrack
- OpenCV
- PyTorch
