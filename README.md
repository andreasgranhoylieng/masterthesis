# Syringe Detection and Volume Estimation System

This repository contains a system for syringe detection, tracking, and volume estimation using computer vision and deep learning.

## Components

1.  **Automatic Annotation (`dino/`)**: Uses GroundingDINO to generate initial annotations for training data.
2.  **Syringe Tracking (`syringe_tracking/`)**: Detects and tracks syringes in video streams using Ultralytics YOLOv11 and ByteTrack.
3.  **Volume Estimation (`volume_estimation/`)**: Estimates syringe volume based on keypoint detection (plunger and barrel tip).

## Features

-   Automatic annotation generation.
-   Real-time syringe detection and tracking.
-   Volume estimation via keypoint analysis.
-   CSV export of volume measurements.
-   Supports video files and webcam streams.
-   Jupyter Notebook interfaces for ease of use.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andreasgranhoylieng/masterthesis.git
    cd masterthesis
    ```

2.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate syringe-ml
    ```
    *(Note: Ensure necessary model weights like GroundingDINO are downloaded. See `dino/README.md` for details).*

3.  **Download Datasets (Optional):**
    If using pre-prepared datasets (e.g., from Roboflow), run the download notebook:
    ```bash
    jupyter notebook download_datasets.ipynb
    ```

## Usage

Each component resides in its own directory with specific instructions:

-   **`dino/`**: Run `main.ipynb` for automatic annotation. Place input images/videos in `dino/images/` or `dino/videos/`.
-   **`syringe_tracking/`**:
    -   Train the detection model: `python train.py` (adjust config as needed).
    -   Run inference: Use `video_inference_od.ipynb` or `webcam_inference_od.ipynb`.
-   **`volume_estimation/`**:
    -   Train the keypoint model: `python train.py` (adjust config as needed).
    -   Run inference and visualization: Use `video_and_webcam_inference.ipynb` or `pose_visualizer.ipynb`.
    -   Data cleaning: `clean_syringe_data.ipynb`.
    -   Tracking animation: `animate_tracking_history.py`.

## Folder Structure

```
.
├── datasets/                 # Downloaded datasets
├── dino/                     # Automatic annotation using GroundingDINO
├── syringe_tracking/         # Syringe detection and tracking
├── volume_estimation/        # Keypoint detection for volume estimation
├── .gitignore
├── download_datasets.ipynb   # Notebook to download datasets
├── environment.yml           # Conda environment definition
└── README.md                 # This file
```