from ultralytics import YOLO

yolo_version = input("Which version of yolo?: ")
dataset_version = input("Enter the dataset version: ")
model_type = input("Enter the model type: ")

# Choose correct extension based on yolo_version
ext = ".yaml" if yolo_version == "12" else ".stl"

# Load the model with the chosen extension
model = YOLO(f"yolo{yolo_version}{model_type}-pose{ext}")

results = model.train(
    data="../datasets/Syringe-volume-estimation-yolo/data.yaml",
    epochs=1200,
    imgsz=1440,
    patience=50,
    workers=32,
    device=[0, 1, 2],
    batch=3 * 4,
    augment=True,
    flipud=0.0,
    fliplr=0.0,
    degrees=170,
    single_cls=True,
    visualize=True,
    pose=50,
    name=f"train-pose{yolo_version}{model_type}-v{dataset_version}",
)