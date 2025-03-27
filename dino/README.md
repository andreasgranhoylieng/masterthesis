# GroundingDINO for Automatic Syringe Annotation

This directory contains the system for automatically generating bounding box annotations for syringes in images or videos using the GroundingDINO zero-shot object detection model.

## Overview

The primary goal is to quickly generate initial annotations based on text prompts (e.g., "syringe"). These annotations can then be refined manually or used directly for training downstream object detection or pose estimation models (like those in the `../syringe_tracking` or `../volume_estimation` directories).

## Features

-   Zero-shot detection based on text prompts.
-   Processes both individual images and video frames.
-   Outputs annotations in Pascal VOC XML format.
-   Includes a utility for video frame extraction.

## File Structure

```
dino/
├── main.ipynb              # Main notebook for running annotation
├── frame_extractor.py      # Utility to extract frames from videos
├── github_files/           # GroundingDINO configs and weights
│   ├── GroundingDINO_SwinB_cfg.py
│   ├── GroundingDINO_SwinT_OGC.py
│   └── groundingdino_swint_ogc.pth # <-- Downloaded weight file
├── images/                 # Input image directory (place images here)
├── videos/                 # Input video directory (place videos here)
├── frames/                 # Output directory for extracted video frames (created by frame_extractor.py)
├── annotations/            # Output directory for generated XML annotations (created by main.ipynb)
└── README.md               # This file
```

## Setup

1.  **Environment:** Ensure you have activated the conda environment specified in the root `environment.yml` file (`conda activate syringe-ml`).
2.  **Download Model Weights:** Download the required GroundingDINO model weights (e.g., `groundingdino_swint_ogc.pth`). Place the `.pth` file inside the `dino/github_files/` directory. Refer to the official GroundingDINO repository if you need download links.
3.  **Prepare Input Data:**
    *   Place images you want to annotate into the `dino/images/` directory.
    *   Place videos you want to annotate into the `dino/videos/` directory.

## Usage

1.  **Extract Frames (Optional, for Videos):**
    If you are using video files, first extract frames using the provided script. Open a terminal in the root directory (`masterthesis/`) and run:
    ```bash
    python dino/frame_extractor.py dino/videos/your_video.mp4 dino/frames/ 1.0
    ```
    *   Replace `your_video.mp4` with your video file name.
    *   The third argument (`1.0`) is the desired frames per second (FPS) to extract. Adjust as needed.
    *   Extracted frames will be saved in the `dino/frames/` directory.

2.  **Run Annotation Notebook:**
    *   Open and run the `dino/main.ipynb` Jupyter Notebook.
    *   Follow the instructions within the notebook.

## Configuration (within `main.ipynb`)

-   **`IMAGE_PATH` / `VIDEO_PATH`:** Set the path to your input images directory (`dino/images/`) or extracted frames directory (`dino/frames/`).
-   **`TEXT_PROMPT`:** Define the object you want to detect (e.g., `"syringe"`).
-   **`BOX_THRESHOLD`, `TEXT_THRESHOLD`:** Adjust detection confidence thresholds if necessary.
-   **`ANNOTATION_SAVE_DIR`:** Specify the directory where XML annotations will be saved (defaults to `dino/annotations/`).

## Output

The system generates annotations in the **Pascal VOC XML** format. Each XML file corresponds to an input image/frame and contains bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`) and the class label (derived from the text prompt) for each detected object. These files are saved in the directory specified by `ANNOTATION_SAVE_DIR`.