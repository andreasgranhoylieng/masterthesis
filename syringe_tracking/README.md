# Syringe Object Detection and Tracking with YOLO

This directory contains scripts and notebooks for training a YOLO model for syringe object detection (OD) and then using the trained model for inference on videos and webcam feeds, including object tracking with ByteTrack.

## Scripts and Notebooks

### 1. Training Script (`train.py`)

This script trains a YOLOv8 model for object detection.

**Functionality:**

*   Prompts the user to input the `dataset_version` and `model_type` (e.g., 'n', 's', 'm', 'l', 'x' for nano, small, medium, large, xlarge YOLOv8 models).
*   Loads a pre-trained YOLOv8 model (e.g., `yolov8n.pt`).
*   Trains the model on a custom dataset specified by `data="../datasets/Syringes/data.yaml"`.
*   Training parameters:
    *   `epochs`: 12000
    *   `imgsz`: 1440
    *   `patience`: 50
    *   `workers`: 32
    *   `device`: [0, 1, 2] (multi-GPU training)
    *   `batch`: 3\*4 (adjust based on GPU memory)
    *   `augment`: True
    *   `name`: Dynamically set based on model type and dataset version (e.g., `train-ODv8n-v1`).

**Usage:**

```bash
python train.py
```

The script will then ask for the dataset version and model type.

### 2. Video Inference Notebook (`video_inference_od.ipynb`)

This Jupyter Notebook performs object detection and tracking on a video file using a trained YOLO model and ByteTrack.

**Key Steps:**

1.  **Environment Setup:** Sets `KMP_DUPLICATE_LIB_OK` to `True` to avoid potential library conflicts.
2.  **Model Loading:** Loads a trained YOLO model (e.g., `runs/detect/train-ODv8x-v8/weights/best.pt`) and moves it to the specified device (e.g., 'mps' for Apple Silicon).
3.  **Inference and Tracking:**
    *   Runs `model.track()` on a source video file (e.g., `'../videos/input_videos/IMG_4732.mov'`).
    *   `show=True`: Displays the video with tracking results.
    *   `tracker="bytetrack_od.yaml"`: Specifies the tracker configuration file.
    *   `save=True`: Saves the output video with tracking results.
    *   The commented-out section shows how to iterate through results and display frames using OpenCV if custom processing is needed.

### 3. Webcam Inference Notebook (`webcam_inference_od.ipynb`)

This Jupyter Notebook performs real-time object detection and pose estimation using two YOLO models (one for detection, one for pose) on a webcam feed.

**Functionality:**

*   Defines a function `run_webcam` that takes paths to a detection model and a pose estimation model, and a boolean `tracking` flag.
*   Loads both YOLO models to the specified device.
*   Opens the default webcam (camera index 0).
*   Sets webcam resolution to 3840x2160 (4K).
*   Continuously captures frames:
    *   Performs object detection (or tracking if `tracking=True`) using the `detect_model`.
    *   Performs pose estimation using the `pose_model`.
    *   Overlays the detection and pose estimation annotations on the frame.
    *   Displays the combined frame.
    *   Exits if the 'q' key is pressed.
*   The `if __name__ == "__main__":` block calls `run_webcam` with example model paths (e.g., `runs/detect/train-ODv8x-v9/weights/best.pt` for detection and `../ultralytics_models/pose/yolov8n-pose.pt` for pose) and `tracking=True`.

### 4. ByteTrack Configuration (`bytetrack_od.yaml`)

This YAML file configures the ByteTrack algorithm used for object tracking.

**Key Parameters:**

*   `tracker_type`: bytetrack
*   `track_high_thresh`: 0.25 (Threshold for the first association)
*   `track_low_thresh`: 0.05 (Threshold for the second association)
*   `new_track_thresh`: 0.25 (Threshold to initialize a new track if a detection doesn't match existing tracks)
*   `track_buffer`: 100 (Number of frames to keep a track alive without new detections)
*   `match_thresh`: 0.9 (Threshold for matching tracks, likely IOU based)
*   `fuse_score`: True (Whether to fuse confidence scores with IOU distances before matching)

## Setup and Running

1.  **Install Dependencies:**
    Ensure `ultralytics` (for YOLO) and `opencv-python` are installed.
    ```bash
    pip install ultralytics opencv-python
    ```
2.  **Dataset:**
    For training, ensure your dataset is structured as expected by YOLO and the `data.yaml` file path in `train.py` (`../datasets/Syringes/data.yaml`) is correct.
3.  **Pre-trained Models:**
    *   For training (`train.py`), YOLOv8 pre-trained weights (`yolov8n.pt`, etc.) will be downloaded automatically by Ultralytics if not present.
    *   For inference notebooks, ensure the paths to your custom trained models (`best.pt`) and any other models (like `yolov8n-pose.pt`) are correct. You might need to download `yolov8n-pose.pt` or other pose models separately from Ultralytics.
4.  **Run Training (if needed):**
    Execute `python train.py` and provide the requested inputs.
5.  **Run Inference:**
    *   Open and run the cells in `video_inference_od.ipynb` or `webcam_inference_od.ipynb` using Jupyter Notebook or JupyterLab.
    *   Modify model paths and video/webcam sources as needed within the notebooks.

## Expected Directory Structure (Illustrative)

```
.
├── train.py
├── video_inference_od.ipynb
├── webcam_inference_od.ipynb
├── bytetrack_od.yaml
├── runs/detect/                    # Created by YOLO training
│   └── train-ODv8x-vX/             # Example training run
│       └── weights/
│           └── best.pt
├── ../datasets/Syringes/           # Dataset location (relative)
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── ../videos/input_videos/         # Example video input location (relative)
│   └── IMG_4732.mov
├── ../ultralytics_models/pose/     # Example pose model location (relative)
│   └── yolov8n-pose.pt