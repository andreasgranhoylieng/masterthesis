# Automatic Annotation with GroundingDINO

This directory contains the system for automatically generating bounding box annotations for syringes in images or videos using the GroundingDINO zero-shot object detection model.

## Purpose

The primary goal is to quickly generate initial annotations that can be refined or used directly for training downstream models (like those in `syringe_tracking` or `volume_estimation`).

## Files

-   `main.ipynb`: The main Jupyter Notebook to run the annotation process. It takes images or video frames as input and outputs annotations in Pascal VOC XML format.
-   `frame_extractor.py`: A utility script to extract frames from video files at a specified frame rate.
-   `github_files/`: Contains GroundingDINO model configuration files (`.py`) and downloaded model weights (`.pth`).
    -   **Note:** Ensure you have downloaded the required weights (e.g., `groundingdino_swint_ogc.pth`) into this directory. Refer to the root `README.md` or the original GroundingDINO repository for download instructions if needed.
-   `images/`: Place input images here.
-   `videos/`: Place input videos here.
-   `frames/`: (Optional) Output directory for frames extracted by `frame_extractor.py`.
-   `annotations/`: Output directory where generated Pascal VOC XML annotations are saved.

## Usage

1.  **Prepare Input:**
    -   Place your images directly into the `images/` directory.
    -   Place your videos into the `videos/` directory.
2.  **Extract Frames (for Videos):**
    If using videos, extract frames first using the utility script:
    ```bash
    python frame_extractor.py videos/your_video.mp4 frames/ 1.0
    ```
    (This example extracts 1 frame per second). Adjust the FPS value as needed. The output frames will be placed in the `frames/` directory (or a subdirectory you specify).
3.  **Run Annotation:**
    -   Open and run the `main.ipynb` notebook.
    -   Configure the input directory (either `images/` or `frames/`) and the text prompt (e.g., "syringe") within the notebook.
    -   The notebook will process the images/frames, perform detection, and save the annotations in XML format to the `annotations/` directory.