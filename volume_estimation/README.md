# Syringe Volume Estimation and Workflow Analysis

This directory contains a suite of Python scripts and Jupyter Notebooks for detecting syringes, estimating their liquid volume using YOLO-based pose estimation, and analyzing a multi-step syringe handling workflow.

## Core Components

### 1. `ActiveZone` Class
Defined in [`demo.py`](volume_estimation/demo.py:15), [`demo_dual_camera.py`](volume_estimation/demo_dual_camera.py:28), [`demo_dual_video.py`](volume_estimation/demo_dual_video.py:29), and [`video_and_webcam_inference.py`](volume_estimation/video_and_webcam_inference.py:14).
*   **Purpose:** Represents a named rectangular area of interest (ROI) within a camera frame. Used to determine if a syringe is within a specific region (e.g., "Table Zone 1", "Arm").
*   **Attributes:**
    *   `name` (str): Unique name for the zone.
    *   `rect` (tuple): Defines the zone as `(x1, y1, x2, y2)`.

### 2. `SyringeVolumeEstimator` Class
Defined in [`demo.py`](volume_estimation/demo.py:46), [`demo_dual_camera.py`](volume_estimation/demo_dual_camera.py:54), [`demo_dual_video.py`](volume_estimation/demo_dual_video.py:55), and [`video_and_webcam_inference.py`](volume_estimation/video_and_webcam_inference.py:42).
*   **Purpose:** Handles syringe detection, tracking (using YOLO with ByteTrack), and volume estimation.
*   **Key Functionality:**
    *   Loads a trained YOLO Pose model (e.g., `runs/pose/train-poseXXX/weights/best.pt`).
    *   Processes video frames to detect syringes and their keypoints (typically 4 keypoints defining the liquid column).
    *   Calculates liquid volume for a list of `possible_diameters_cm` based on the detected keypoints. The volume is estimated by calculating the height of the liquid column in pixels, converting it to centimeters using the assumed diameter as a scale, and then applying the cylinder volume formula.
    *   Identifies if a syringe is within any defined `ActiveZone` based on an `area_threshold` (percentage of syringe bounding box overlapping with the zone).
    *   Draws annotations on frames: bounding boxes, keypoints, volume tables per syringe, FPS counter, and active zone highlights.
    *   Logs raw detection and volume data to a CSV file.
*   **Configuration:**
    *   `model_path`: Path to the YOLO Pose model.
    *   `possible_diameters_cm`: List of syringe diameters (in cm) for which volumes will be estimated.
    *   `active_zones`: A list of `ActiveZone` objects.
    *   `area_threshold`: Minimum overlap ratio for a syringe to be considered inside an active zone.
    *   `device_preference` (in `demo_dual_*.py`): Preferred compute device ("cuda", "mps", "cpu", or `None` for auto-detection).
    *   `DEBUG_MODE` flag (in `demo_dual_*.py`): Enables detailed print statements for debugging.

### 3. `SyringeTestWorkflow` Class
Defined in [`demo_dual_camera.py`](volume_estimation/demo_dual_camera.py:278) and [`demo_dual_video.py`](volume_estimation/demo_dual_video.py:286).
*   **Purpose:** Manages and validates a predefined syringe handling workflow. This version uses a simplified logic without timeouts and employs persistent volume checking after a syringe is inserted into a target zone.
*   **States:**
    *   `STATE_IDLE`: Waiting for a syringe to be picked up.
    *   `STATE_SYRINGE_PICKED`: A syringe has been picked from a table zone and is being handled.
    *   `STATE_SYRINGE_INSERTED`: The syringe has been detected in a target zone (e.g., manikin's arm).
*   **Workflow Logic:**
    *   Initializes by scanning syringes on "table zones" (seen by one camera/video).
    *   Detects a "pickup" when a syringe disappears from a table zone.
    *   Detects an "insertion" when a syringe appears in a "target zone" (seen by another camera/video).
    *   Persistently checks for volume once inserted until a valid volume is determined or the syringe is returned.
    *   Detects a "return" when a syringe reappears in a table zone.
    *   Logs errors based on:
        *   `ERROR_WRONG_SYRINGE`: Picked from an incorrect starting table zone.
        *   `ERROR_WRONG_VOLUME`: Measured volume (for the `correct_syringe_diameter`) is outside `target_volume_ml` ± `volume_tolerance_ml`.
        *   `ERROR_WRONG_TARGET`: Inserted into an incorrect target zone.
        *   `ERROR_MULTI_ACTIVE`: Multiple syringes appear to be picked up simultaneously.
        *   `ERROR_PREMATURE_RETURN`: Syringe returned to table before insertion.
        *   `ERROR_UNEXPECTED_INSERT`/`RETURN`: Syringe appears in a target/table zone when not expected by the current state.
*   **Configuration:**
    *   `table_zone_names`, `target_zone_names`: Names of zones designated as table or target areas.
    *   `correct_starting_zone`: Expected table zone for pickup.
    *   `correct_syringe_diameter`: The true diameter of the syringe being used in the test.
    *   `target_volume_ml`, `volume_tolerance_ml`: Expected volume and its tolerance.
    *   `correct_target_zone`: Expected target zone for insertion.
    *   `log_file_path`: Path for the detailed workflow log.

## Scripts and Notebooks

### Training
*   **[`train.py`](volume_estimation/train.py:1):**
    *   **Purpose:** Trains a YOLOv8 (or v12 based on input) pose estimation model for syringes.
    *   **Usage:** Prompts the user for YOLO version, dataset version, model type (n, s, m, l, x), and whether to resume training.
    *   **Configuration:**
        *   Dataset path: `../datasets/Syringe-volume-estimation-yolo/data.yaml`
        *   Training arguments (epochs, image size, patience, augmentation, etc.) are hardcoded but can be modified.

### Inference and Demonstration
*   **[`demo.py`](volume_estimation/demo.py:1):**
    *   **Purpose:** General-purpose script to run syringe volume estimation on a webcam feed or a video file using the `SyringeVolumeEstimator`. Includes a basic workflow manager (`SyringeTestWorkflow`).
    *   **Configuration (User Editable Section):**
        *   `YOLO_MODEL_PATH`
        *   `POSSIBLE_SYRINGE_DIAMETERS_CM`
        *   `FRAME_WIDTH`, `FRAME_HEIGHT` (for zone definition reference)
        *   `TABLE_ZONE_NAMES`, `TARGET_ZONE_NAMES`, `ZONE_DEFINITIONS` (list of `ActiveZone` objects)
        *   Workflow parameters: `CORRECT_STARTING_ZONE`, `CORRECT_SYRINGE_DIAMETER_CM`, `TARGET_VOLUME_ML`, etc.
        *   `INPUT_SOURCE` ('webcam' or 'video'), `VIDEO_PATH`
        *   `SAVE_OUTPUT_VIDEO`, `RAW_CSV_PATH`, `WORKFLOW_LOG_PATH`
*   **[`video_and_webcam_inference.py`](volume_estimation/video_and_webcam_inference.py:1):**
    *   **Purpose:** A simpler script for running the `SyringeVolumeEstimator` on webcam or video. Focuses on detection and volume estimation without the complex `SyringeTestWorkflow`.
    *   **Configuration:** Active zones can be defined and passed to `SyringeVolumeEstimator`.
    *   **Usage:** Can be run with `input_source='webcam'` or `input_source='video'` with a `video_path`.
*   **[`demo_dual_camera.py`](volume_estimation/demo_dual_camera.py:1):**
    *   **Purpose:** Implements the `SyringeTestWorkflow` using two live webcams: one observing the "syringes" (table zones) and one observing the "manikin" (target zones).
    *   **Configuration (User Editable Section):** Similar to `demo.py` but includes `MANIKIN_CAMERA_INDEX`, `SYRINGES_CAMERA_INDEX`, and separate zone definitions for manikin and syringe views.
*   **[`demo_dual_video.py`](volume_estimation/demo_dual_video.py:1):**
    *   **Purpose:** Implements the `SyringeTestWorkflow` using two synchronized video files (manikin and syringes perspectives). Assumes videos start simultaneously.
    *   **Configuration (User Editable Section):** Similar to `demo_dual_camera.py` but takes `MANIKIN_VIDEO_PATH` and `SYRINGES_VIDEO_PATH` instead of camera indices.

### Utilities
*   **[`record_video.py`](volume_estimation/record_video.py:1):**
    *   **Purpose:** Records synchronized video streams from two cameras.
    *   **Configuration:** `camera_index_1`, `camera_index_2`, `desired_capture_width`, `desired_capture_height`, `desired_fps`, `output_dir`.
    *   **Output:** Saves two `.mp4` files, timestamped and named by camera index, resolution.
*   **[`visualize_features.py`](volume_estimation/visualize_features.py:1):**
    *   **Purpose:** Visualizes CNN feature map activations from a PyTorch model (`.pt` file).
    *   **Usage:** Run from CLI with arguments: `--model_path`, `--image_path`, `--output_dir`, `--img_size`, `--max_channels`.
    *   **Functionality:** Registers forward hooks on model layers (attempts to find a `nn.Sequential` block), performs a forward pass with an input image, and saves the captured feature maps as SVG files.

### Analysis and Metrics
*   **[`metric_calculation.ipynb`](volume_estimation/metric_calculation.ipynb:1):**
    *   **Purpose:** Processes a batch of videos (found in `videos/` subdirectories). For each video:
        1.  Runs the `SyringeVolumeEstimator` (from `video_and_webcam_inference.py`).
        2.  Reads the temporary `syringe_data.csv` generated by the estimator.
        3.  Extracts actual volume, diameter, and zoom level from the video filename using regex.
        4.  Calculates statistics (min, max, mean, std, median, SEM, CV) for the estimated volumes for the correct diameter.
        5.  Appends these metrics to a `final_dataframe`.
    *   **Output:** Saves `syringe_volume_estimations_and_metrics.csv`.
*   **[`metrics_comparison.ipynb`](volume_estimation/metrics_comparison.ipynb:1):**
    *   **Purpose:** Compares the performance of different models (e.g., "Nano (N)" vs "X" model variants) based on the CSV files generated by `metric_calculation.ipynb` (e.g., `syringe_volume_estimations_and_metrics_N.csv`, `syringe_volume_estimations_and_metrics_X.csv`).
    *   **Functionality:**
        1.  Loads and combines data from different model CSVs.
        2.  Calculates error metrics: `error (ml)`, `abs_error (ml)`, `percent_error (%)`, `abs_percent_error (%)`.
        3.  Generates various plots:
            *   Mean estimated volume vs. actual volume.
            *   Distribution of absolute error.
            *   Distribution of Coefficient of Variation (CV).
            *   Analysis of estimations for 0ml (empty) syringes.
            *   Impact of zoom level on errors and CV.
    *   **Output:** Displays plots and prints summary statistics. Saves plots as PNG files.
*   **`recordings/` directory:**
    *   Contains example CSVs (`metrics_summary_yolo_analysis.csv`, `overview.csv`) and notebooks (`group_recordings.ipynb`, `overview_analysis.ipynb`) for further specialized analysis of recorded data.

## Tracker Configuration
*   **[`bytetrack_pose.yaml`](volume_estimation/bytetrack_pose.yaml:1) (and referred to as `bytetrack.yaml` in scripts):**
    *   Configuration file for the ByteTrack algorithm used by YOLO for object tracking.
    *   Specifies parameters like `tracker_type`, `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`.

## Data Directories
*   **`videos/`:** Contains subfolders (`1x/`, `3x/`) with `.mov` video files used for testing and metric calculation. Filenames typically encode diameter and actual volume (e.g., `1.0-3ml.mov`).
*   **`recordings/`:** Default output directory for `record_video.py`. Also contains sub-notebooks and CSVs for analyzing recorded sessions.
*   **`saved_csv_files/`:** Contains example CSV outputs from volume estimation runs (e.g., `syringe_volume_estimations_and_metrics_N.csv`).
*   **`saved_error_files/`:** Contains example workflow log files (`.txt`) generated by the `SyringeTestWorkflow` class.

## General Setup and Usage Notes
1.  **Model Paths:** Ensure the `YOLO_MODEL_PATH` in the demo scripts and `model_path` in `SyringeVolumeEstimator` point to your trained YOLO Pose model (usually a `best.pt` file).
2.  **Dataset for Training:** The `train.py` script expects the dataset at `../datasets/Syringe-volume-estimation-yolo/data.yaml`.
3.  **Dependencies:** Ensure `ultralytics`, `torch`, `torchvision`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `seaborn` are installed.
4.  **Active Zones:** Carefully define `ActiveZone` coordinates based on your camera setup and frame resolution.
5.  **Workflow Parameters:** When using workflow scripts (`demo_dual_*.py`, `demo.py`), accurately set parameters like `CORRECT_STARTING_ZONE`, `CORRECT_SYRINGE_DIAMETER_CM`, `TARGET_VOLUME_ML`, etc., to match your test scenario.
6.  **Camera Indices/Video Paths:** For scripts involving camera input or video files, verify that camera indices or file paths are correctly specified.