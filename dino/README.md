# GroundingDINO Automatic Annotation System

This component handles automatic annotation of syringe images/videos using GroundingDINO zero-shot object detection.

## Key Components

- `main.ipynb`: Main notebook for running inference and generating Pascal VOC annotations
- `frame_extractor.py`: Utility for extracting frames from videos at specified FPS
- `github_files/`: Contains model configs and weights:
  - `GroundingDINO_SwinT_OGC.py`: Swin-Tiny model config 
  - `GroundingDINO_SwinB_cfg.py`: Swin-Base model config
  - `groundingdino_swint_ogc.pth`: Pretrained weights

## Directory Structure
```
dino/
├── annotations/       # Generated XML annotations
├── frames/            # Extracted video frames
├── images/            # Input images for annotation
├── videos/            # Input videos for processing
├── main.ipynb         # Annotation pipeline
└── frame_extractor.py # Frame extraction utility
```

## Usage

1. Place input media in `images/` or `videos/`
2. For videos, extract frames first:
```bash
python frame_extractor.py input.mp4 frames/ 1.0 # Extract 1 FPS
```
3. Run `main.ipynb` to:
   - Detect syringes using text prompts
   - Generate Pascal VOC XML annotations
   - Create annotated preview images
   - Output to `annotations/` and `annotated_images/`

## Dependencies
- GroundingDINO
- OpenCV
- PyTorch
- Jupyter Notebook
