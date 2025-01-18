from ultralytics import YOLO

# Load a model
model = YOLO("../ultralytics_models/pose/yolo11n-pose.pt")
results = model.train(
    data="../datasets/Syringe-volume-estimation/data.yaml", epochs=10_000, imgsz=640, patience=100
)