from ultralytics import YOLO

dataset_version = input("Enter the dataset version: ")
model_type = input("Enter the model type: ")

# Load a pre-trained YOLO11 model
model = YOLO(f"yolo11{model_type}.pt")

# Train the model on your dataset
results = model.train(
    data="../datasets/Syringes/data.yaml",
    epochs=12000,
    imgsz=1440,
    patience=50,
    workers=32,
    device=[0, 1, 2],
    batch=3*4,
    augment=True,
    name=f"train-OD11{model_type}-v{dataset_version}",
)
