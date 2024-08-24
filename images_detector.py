from ultralytics import YOLO
import numpy

model = YOLO("yolov8n.pt", "v8")

detection_output = model.predict(source="Images/image3.webp", conf=0.25, save=True)

print(detection_output)

print(detection_output[0].numpy())