# Syringe Detection and Volume Estimation System

This repository contains a comprehensive system for syringe detection, tracking, and volume estimation using computer vision and deep learning. The system consists of three main components:

1. **Automatic Annotation with GroundingDINO** - For generating training data
2. **Keypoint Detection for Volume Estimation** - For measuring syringe plunger displacement
3. **Syringe Tracking** - For tracking multiple syringes in video streams

## Features

- 🎯 Automatic annotation of syringe images using GroundingDINO
- 📏 Precise volume estimation using keypoint detection
- 📹 Real-time syringe tracking in video streams
- 📊 CSV export of volume measurements over time
- 🎥 Webcam and video file support
- 📦 Easy-to-use Jupyter Notebook interfaces

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/andreasgranhoylieng/masterthesis.git
   cd syringe-ml
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate syringe-ml
   ```

3. Install GroundingDINO weights:
   ```bash
   cd dino
   mkdir -p github_files
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O github_files/groundingdino_swint_ogc.pth
   cd ..
   ```

## Usage

### 1. Automatic Annotation with GroundingDINO
- Place your images or videos in `dino/images/` or `dino/videos/`
- Run `dino/main.ipynb` to generate annotations
- Annotations will be saved in Pascal VOC format
- These annotations can be uploaded to Roboflow

### 2. Volume Estimation with Keypoint Detection
1. Download datasets:
   ```bash
   jupyter notebook download_datasets.ipynb
   ```
2. Train the model:
   ```bash
   jupyter notebook volume_estimation/train.ipynb
   ```
3. Run inference:
   - For video files: `volume_estimation/pose_video_visualizer.ipynb`
   - For webcam: `volume_estimation/webcam_inference_pose.ipynb`

### 3. Syringe Tracking
1. Train the detection model:
   ```bash
   jupyter notebook syringe_tracking/train.ipynb
   ```
2. Run tracking:
   - For video files: `syringe_tracking/video_inference_od.ipynb`
   - For webcam: `syringe_tracking/webcam_inference_od.ipynb`

## Folder Structure

```
.
├── dino/                     # GroundingDINO automatic annotation
│   ├── images/               # Input images for annotation
│   ├── videos/               # Input videos for annotation
│   ├── annotations/          # Generated Pascal VOC annotations from images or videos
│   ├── frames/               # Frames extracted from videos
│   ├── github_files/         # GroundingDINO weights and configs
│   └── main.ipynb            # Annotation notebook
├── volume_estimation/        # Volume estimation system
│   ├── train.ipynb           # Training notebook
│   ├── pose_video_visualizer.ipynb # Video processing
│   └── webcam_inference_pose.ipynb # Webcam inference
├── syringe_tracking/         # Syringe tracking system
│   ├── train.ipynb           # Training notebook
│   ├── video_inference_od.ipynb # Video tracking
│   └── webcam_inference_od.ipynb # Webcam tracking
├── datasets/                 # Dataset storage from Roboflow
├── environment.yml           # Conda environment
└── README.md                 # This file
```
