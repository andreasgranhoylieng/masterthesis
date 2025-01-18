from roboflow import Roboflow


# Volume estimation dataset
rf = Roboflow(api_key="WhO3q8dLKIclIg7IxQpe")
project = rf.workspace("mastersmedical").project("syringe-volume-estimation")
version = project.version(1)
dataset = version.download("yolov8", location="datasets/Syringe-volume-estimation")
                