# Data Annotation Tools

This directory (`data_annotation/`) contains tools and scripts for generating annotations for images or videos. This can include automatic, semi-automatic, or manual annotation methods.

## Overview

The primary goal of this component is to prepare labeled data for training computer vision models. This might involve using pre-trained models for initial suggestions (like zero-shot detectors), or providing interfaces for manual labeling. The output annotations are typically used by downstream components like `../object_detection_tracking/` or `../downstream_task/`.

## Features

-   Supports various annotation methods (e.g., bounding boxes, segmentation masks, keypoints).
-   Can process individual images and/or video frames.
-   Outputs annotations in common formats (e.g., COCO JSON, Pascal VOC XML, YOLO TXT).
-   May include utilities for data preprocessing, like video frame extraction or image augmentation.

## File Structure (Example)

```
data_annotation/
├── main_annotation_notebook.ipynb  # Main notebook for running/managing annotation tasks
├── annotation_script.py          # Example script for an automated annotation process
├── utils/                          # Utility scripts (e.g., frame_extractor.py, format_converter.py)
│   ├── frame_extractor.py
│   └── format_converter.py
├── configs/                        # Configuration files for annotation tools or models
│   └── annotation_config.yaml
├── models/                         # Any models used for pre-annotation (e.g., GroundingDINO weights)
│   └── model_weights.pth
├── input_data/                     # Directory for raw images/videos to be annotated
│   ├── images/
│   └── videos/
├── extracted_frames/               # Output for extracted video frames (if applicable)
├── output_annotations/             # Directory for generated annotation files
└── README.md                       # This file
```

## Setup

1.  **Environment:** Ensure you have activated the conda environment specified in the root `environment.yml` file (e.g., `conda activate your-env-name`).
2.  **Download Models/Dependencies:** If using specific pre-trained models (e.g., for zero-shot detection) or external tools, download them and place them in the appropriate directory (e.g., `data_annotation/models/`). Refer to specific tool documentation for details.
3.  **Prepare Input Data:**
    *   Place images to be annotated into `data_annotation/input_data/images/`.
    *   Place videos to be annotated into `data_annotation/input_data/videos/`.

## Usage

1.  **Pre-processing (e.g., Frame Extraction for Videos):**
    If working with videos, you might need to extract frames first.
    ```bash
    # Example:
    python data_annotation/utils/frame_extractor.py data_annotation/input_data/videos/your_video.mp4 data_annotation/extracted_frames/ 1.0
    ```
    *   Replace `your_video.mp4` with your video file name.
    *   Adjust FPS as needed.
    *   Extracted frames will be saved in `data_annotation/extracted_frames/`.

2.  **Run Annotation Process:**
    *   Use the main script or notebook, e.g., `data_annotation/main_annotation_notebook.ipynb` or `python data_annotation/annotation_script.py`.
    *   Follow the instructions within the chosen script/notebook.

## Configuration (Example, within a script or notebook)

-   **`INPUT_IMAGE_DIR` / `INPUT_VIDEO_DIR`:** Path to your input data.
-   **`OUTPUT_ANNOTATION_DIR`:** Directory to save generated annotations.
-   **`ANNOTATION_FORMAT`:** Desired output format (e.g., "coco", "voc", "yolo").
-   **`MODEL_CONFIG_PATH` (if applicable):** Path to configuration for any models used in annotation.
-   **`LABEL_MAP` (if applicable):** Definition of class labels.

## Output

The system generates annotation files in the specified format (e.g., Pascal VOC XML, COCO JSON). These files typically contain information about object locations (bounding boxes, masks), class labels, and other relevant metadata. They are saved in the directory specified by `OUTPUT_ANNOTATION_DIR`.