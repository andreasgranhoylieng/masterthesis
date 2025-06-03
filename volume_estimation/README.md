# Downstream Task Application

This directory (`downstream_task/`) houses the specific application logic that utilizes the outputs from preceding computer vision modules, such as `../data_annotation/` and `../object_detection_tracking/`. This could be anything from quantitative analysis (e.g., volume estimation, counting) to qualitative assessment (e.g., action recognition, anomaly detection).

## Purpose

The goal of this module is to take processed visual information (like object detections, tracks, or segmentations) and apply further logic or models to achieve a specific project objective. For example, if the `object_detection_tracking` module identifies and tracks "syringes," this `downstream_task` module might then use keypoint detection on those syringes to estimate their volume.

## Files (Example Structure)

-   `train_task_model.py`: Script for training any specific models required for this task (e.g., a keypoint detector, a classifier).
-   `run_application_video.ipynb`: Jupyter Notebook for applying the downstream task logic to video files.
-   `run_application_webcam.ipynb`: Jupyter Notebook for real-time application using a webcam.
-   `task_specific_utils.py`: Utility functions specific to this downstream task (e.g., geometric calculations, data transformations).
-   `configs/`: Configuration files for the task-specific models or logic.
    -   `task_model_config.yaml`
    -   `application_params.yaml`
-   `trained_task_models/`: Directory for storing trained models specific to this task.
-   `results/`: Directory for saving outputs of the application (e.g., CSV files, images, reports).

## Usage

1.  **Prepare Input Data/Models:**
    Ensure that the necessary inputs are available. This might include:
    *   Outputs from the `../object_detection_tracking/` module (e.g., bounding boxes, track IDs).
    *   Specific datasets for training task-specific models (e.g., images annotated with keypoints).
    *   Pre-trained models from previous stages.

2.  **Train Task-Specific Model (If Applicable):**
    If your downstream task requires its own model (e.g., a pose estimator, a classifier), train it using `train_task_model.py`.
    ```bash
    python train_task_model.py --data_path path/to/task_data --config configs/task_model_config.yaml
    ```
    Adjust parameters as needed. Trained models are typically saved in `trained_task_models/`.

3.  **Run the Application:**
    -   **Video File:** Open and run `run_application_video.ipynb`. Configure paths to input video, models from previous stages (e.g., object detector weights), and any task-specific models.
    -   **Webcam:** Open and run `run_application_webcam.ipynb`. Set relevant model paths and webcam index.
    The notebooks will typically load necessary models, process the input, apply the downstream logic, and display or save the results.

4.  **Analyze Results:**
    Outputs (e.g., measurements, classifications, visualizations) are often saved in the `results/` directory for further analysis.

## Configuration

-   **Task-Specific Models:** Configuration for any models trained or used within this module (e.g., architecture, hyperparameters) is managed via config files (e.g., `configs/task_model_config.yaml`) or arguments to scripts/notebooks.
-   **Application Parameters:** Settings for the overall application logic (e.g., thresholds, input/output paths, specific calculation parameters) can be set in notebooks or a dedicated config file like `configs/application_params.yaml`.
-   **Integration:** Ensure that data formats and interfaces are compatible with the upstream modules (`../object_detection_tracking/`).
