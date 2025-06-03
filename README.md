# Syringe Analysis and Workflow Automation Project

This repository contains a collection of tools and experiments for syringe detection, volume estimation, tracking, and workflow automation using computer vision and machine learning techniques.

## Project Structure

The project is organized into the following main directories:

*   **[`dino/`](dino/):** Contains scripts for object detection using GroundingDINO.
    *   [`frame_extractor.py`](dino/frame_extractor.py:1): Extracts frames from videos.
    *   [`main.ipynb`](dino/main.ipynb:1): Performs object detection on extracted frames and saves annotations in Pascal VOC XML format.
    *   See [`dino/README.md`](dino/README.md:1) for more details.
*   **[`syringe_tracking/`](syringe_tracking/):** Focuses on training YOLO models for syringe object detection and tracking.
    *   [`train.py`](syringe_tracking/train.py:1): Trains YOLOv8 models for object detection.
    *   [`video_inference_od.ipynb`](syringe_tracking/video_inference_od.ipynb:1): Performs object detection and tracking on videos using a trained model and ByteTrack.
    *   [`webcam_inference_od.ipynb`](syringe_tracking/webcam_inference_od.ipynb:1): Performs real-time object detection and pose estimation from a webcam.
    *   [`bytetrack_od.yaml`](syringe_tracking/bytetrack_od.yaml:1): Configuration for the ByteTrack tracker.
    *   See [`syringe_tracking/README.md`](syringe_tracking/README.md:1) for more details.
*   **[`volume_estimation/`](volume_estimation/):** Contains scripts and notebooks for estimating syringe volume using YOLO Pose and analyzing clinical workflows.
    *   Demonstration scripts (`demo.py`, `demo_dual_camera.py`, `demo_dual_video.py`): Showcase volume estimation and workflow validation using single or dual camera/video inputs.
    *   [`train.py`](volume_estimation/train.py:1): Trains YOLO Pose models for syringe keypoint detection.
    *   [`video_and_webcam_inference.py`](volume_estimation/video_and_webcam_inference.py:1): Simpler inference script for volume estimation.
    *   [`record_video.py`](volume_estimation/record_video.py:1): Utility to record synchronized videos from two cameras.
    *   [`visualize_features.py`](volume_estimation/visualize_features.py:1): Tool to visualize CNN feature maps.
    *   Analysis notebooks (`metric_calculation.ipynb`, `metrics_comparison.ipynb`): Process experimental data and compare model performances.
    *   See [`volume_estimation/README.md`](volume_estimation/README.md:1) for more details.
*   **[`datasets/`](datasets/):** Intended location for datasets. Contains a `.gitkeep` file. Datasets are downloaded here by [`download_datasets.ipynb`](download_datasets.ipynb:1).
    *   `Syringe-volume-estimation-yolo/`: Dataset for YOLO Pose estimation.
    *   `Syringes/`: Dataset for general syringe object detection.

## Root Directory Files

*   **[`download_datasets.ipynb`](download_datasets.ipynb:1):** Jupyter Notebook to download datasets from Roboflow using an API key (requires a `.env` file with `ROBOFLOW_API_KEY`).
    *   Downloads "syringe-volume-estimation" (version 13) for YOLOv8 pose.
    *   Downloads "syringe-tracker" (version 4) for YOLOv11 object detection.
*   **[`environment.yml`](environment.yml:1):** Conda environment file listing project dependencies. This allows for reproducible environments.
*   **[`id_generator.ipynb`](id_generator.ipynb:1):** Jupyter Notebook to generate random 4-digit IDs and three distinct tasks for a simulated medical scenario. Each task involves a body part, a syringe type, and a dose.
*   **[`pyproject.toml`](pyproject.toml:1):** Configuration file for Python project tools, primarily `ruff` for linting and formatting. It defines excluded directories, line length, target Python version, and enabled linting rules.
*   **[`.gitignore`](.gitignore:1):** Specifies intentionally untracked files that Git should ignore (e.g., `runs/`, `datasets/`, `*.mp4`, `*.csv`, `__pycache__/`).

## General Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andreasgranhoylieng/masterthesis.git
    cd masterthesis
    ```
2.  **Create Conda Environment:**
    It's highly recommended to use the provided Conda environment file to ensure all dependencies are correctly installed.
    ```bash
    conda env create -f environment.yml
    conda activate syringe-ml
    ```
3.  **Download Datasets:**
    *   Create a `.env` file in the root directory with your Roboflow API key:
        ```
        ROBOFLOW_API_KEY=your_api_key_here
        ```
    *   Run the [`download_datasets.ipynb`](download_datasets.ipynb:1) notebook to download the necessary datasets into the `datasets/` folder.
4.  **Pre-trained Models:**
    *   The scripts generally expect pre-trained models (e.g., `best.pt` files from YOLO training runs) to be present in specific `runs/.../weights/` directories or specified paths.
    *   You may need to train models first using the provided training scripts (e.g., [`syringe_tracking/train.py`](syringe_tracking/train.py:1), [`volume_estimation/train.py`](volume_estimation/train.py:1)) or download pre-trained weights if available.

## Usage

Refer to the README files within each subdirectory (`dino/`, `syringe_tracking/`, `volume_estimation/`) for specific instructions on how to run the scripts and notebooks therein.

### Linting and Formatting
This project uses `ruff` for linting and formatting, configured via [`pyproject.toml`](pyproject.toml:1).
You can run `ruff check .` to lint and `ruff format .` to format the code.
`nbqa` can be used to run `ruff` on Jupyter notebooks: `nbqa ruff .`

## Key Technologies
*   Python 3.9
*   PyTorch
*   Ultralytics YOLO (v8, v11, v12 for pose and object detection)
*   OpenCV
*   GroundingDINO
*   Pandas, NumPy, Matplotlib, Seaborn (for data analysis and visualization)
*   Roboflow (for dataset management)
*   Conda (for environment management)
*   Ruff (for linting/formatting)

This project forms the basis of a master's thesis focused on applying computer vision techniques to medical (syringe-related) tasks.