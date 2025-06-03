# GroundingDINO Object Detection and Annotation

This directory contains scripts for extracting frames from videos and performing object detection using the GroundingDINO model. The primary use case demonstrated is detecting syringes in images and saving annotations in Pascal VOC XML format.

## Scripts

### 1. Frame Extractor (`frame_extractor.py`)

This script extracts frames from a video file at a user-specified frames-per-second (FPS) rate.

**Usage:**

```bash
python frame_extractor.py <video_path> <output_dir> <fps>
```

**Arguments:**

*   `<video_path>`: Path to the input video file.
*   `<output_dir>`: Directory where the extracted frames will be saved.
*   `<fps>`: The desired number of frames to extract per second.

The script will create the output directory if it doesn't exist and save frames as PNG images (e.g., `frame_00000.png`, `frame_00001.png`, ...).

### 2. Main Annotation Notebook (`main.ipynb`)

This Jupyter Notebook uses a pre-trained GroundingDINO model to detect objects in images based on a text prompt. It then saves the bounding box annotations in Pascal VOC XML format and also saves the images with the detected bounding boxes drawn on them.

**Key Steps:**

1.  **Configuration:**
    *   `config_path`: Path to the GroundingDINO model configuration file (e.g., `github_files/GroundingDINO_SwinT_OGC.py`).
    *   `weights_path`: Path to the GroundingDINO model weights file (e.g., `github_files/groundingdino_swint_ogc.pth`).
    *   `image_directory`: Directory containing the images to be processed (default: `"frames"`). This is typically the output directory from `frame_extractor.py`.
    *   `annotation_directory`: Directory where the Pascal VOC XML annotation files will be saved (default: `"annotations"`).
    *   `annotated_image_directory`: Directory where the images with drawn annotations will be saved (default: `"annotated_images"`).
    *   `text_prompt`: The text description of the object to detect (e.g., `"all single syringes"`).
    *   `box_threshold`: Confidence threshold for object detection bounding boxes.
    *   `text_threshold`: Confidence threshold for text-based filtering.

2.  **Directory Reset:** The script includes a function to reset (delete and recreate) the `annotation_directory` and `annotated_image_directory` before processing.

3.  **Model Loading:** Loads the specified GroundingDINO model.

4.  **Image Processing Loop:**
    *   Iterates through all valid image files (JPG, PNG, etc.) in the `image_directory`.
    *   For each image:
        *   Runs the GroundingDINO model to predict bounding boxes based on the `text_prompt`.
        *   Converts the detected bounding boxes to Pascal VOC format.
        *   Saves the annotation as an XML file in the `annotation_directory`.
        *   Draws the bounding boxes on the original image and saves it in the `annotated_image_directory`.

**Dependencies (主なもの):**

*   `torch`
*   `torchvision`
*   `opencv-python` (`cv2`)
*   `groundingdino` (and its dependencies)
*   `tqdm`
*   `jupyter` (to run the notebook)

**Expected Directory Structure:**

Before running `main.ipynb`, you should have:

```
.
├── github_files/
│   ├── GroundingDINO_SwinT_OGC.py
│   └── groundingdino_swint_ogc.pth
├── frames/                  # Contains images extracted by frame_extractor.py
│   ├── frame_00000.png
│   └── ...
└── main.ipynb
```

After running `main.ipynb`, the following directories will be created/populated:

```
.
├── annotations/             # Contains Pascal VOC XML annotations
│   ├── frame_00000.xml
│   └── ...
├── annotated_images/        # Contains images with drawn bounding boxes
│   ├── frame_00000_annotated.jpg
│   └── ...
...
```

## Setup and Running

1.  **Clone the Repository (if not already done):**
    Ensure you have the necessary model files (`GroundingDINO_SwinT_OGC.py`, `groundingdino_swint_ogc.pth`) in the `github_files` directory. These might need to be downloaded separately if not included.
2.  **Install Dependencies:**
    Make sure all Python dependencies are installed. You might use a `requirements.txt` or `environment.yml` if provided at the root of the project.
3.  **Extract Frames (Optional but Recommended):**
    Use `frame_extractor.py` to get images from your video.
    ```bash
    python dino/frame_extractor.py path/to/your/video.mp4 dino/frames 1
    ```
4.  **Run the Annotation Notebook:**
    Open and run the cells in `dino/main.ipynb` using Jupyter Notebook or JupyterLab. Adjust the configuration parameters at the beginning of the notebook as needed.