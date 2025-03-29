from ultralytics import YOLO

def str_to_bool(s: str) -> bool:
    return s.lower() in ["yes", "true", "1"]

yolo_version = input("Which version of YOLO?: ")
dataset_version = input("Enter the dataset version: ")
model_type = input("Enter the model type (e.g., 'n', 's', 'm', 'l', 'x'): ")
resume_input = input("Resume training? (True/False): ")

resume = str_to_bool(resume_input)

# Pick file extension based on model_type
ext = ".yaml" if model_type.lower() == "n" else ".pt"

# Load the model
if resume:
    model = YOLO(f"runs/pose/train-pose{yolo_version}{model_type}-v{dataset_version}/weights/last.pt")
else:
    model = YOLO(f"yolo{yolo_version}{model_type}-pose{ext}")

# Define training config
train_args = {
    "data": "../datasets/Syringe-volume-estimation-yolo/data.yaml",
    "epochs": 10000,
    "imgsz": 1440,
    "patience": 50,
    "workers": 32,
    "device": [0, 1, 2],
    "batch": 3 * 4,
    "augment": True,
    "flipud": 0.0,
    "fliplr": 0.0,
    "degrees": 170,
    "single_cls": True,
    "visualize": True,
    "name": f"train-pose{yolo_version}{model_type}-v{dataset_version}",
}

if resume:
    train_args["resume"] = True

# Train the model
results = model.train(**train_args)