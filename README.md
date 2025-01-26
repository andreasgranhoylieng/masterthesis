# Master's Thesis

## YOLO Pose Usage

Follow the steps below to use YOLO Pose from scratch:

### 1. Download the Dataset
- Run the `download_datasets.ipynb` notebook to download and format the dataset.
- The dataset will be saved in the `dataset/` folder.
- Update the `data.yaml` file:
  - Set the paths for `train`, `test`, and `valid` to the full paths of their respective folders.

### 2. Train the Model
- Open and run the `volume_estimation_yolo/train.ipynb` notebook with your desired parameters to train the model.

### 3. Evaluate the Model
- Test the model's performance by running `volume_estimation_yolo/pose_visualizer.ipynb`.

### 4. Live Inference on Camera
- Use `volume_estimation_yolo/webcam_inference.ipynb` to run the model live on a camera feed.