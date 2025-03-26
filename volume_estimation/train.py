from ultralytics import YOLO

dataset_version = input("Enter the dataset version: ")
model_type = input("Enter the model type: ")

# Load a model
model = YOLO(f"yolo11{model_type}-pose.pt")
results = model.train(
    data="../datasets/Syringe-volume-estimation-yolo/data.yaml",
    epochs=1200,
    imgsz=1440,
    patience=50,
    workers=32,
    device=[0, 1, 2],
    batch=3*4,
    augment=True,
    flipud=0.0,
    fliplr=0.0,
    degrees=170,
    single_cls=True,
    visualize=True,
    pose=50,
    name=f"train-pose11{model_type}-v{dataset_version}",
)
