from ultralytics import YOLO

model = YOLO("runs/pose/train5/weights/best.pt")  # load a custom model

# Predict with the model
results = model("datasets/Syringe-volume-estimation/test/images/IMG_4314_jpeg.rf.26f801d9e87fd89294b0c2231895d40b.jpg")  # predict on an image
results.show()  # display the image