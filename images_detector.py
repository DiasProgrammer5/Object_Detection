from ultralytics import YOLO
import cv2
import glob

model = YOLO("yolov8n.pt", "v8")

image_name = "image3.webp"

detection_output = model.predict(source=f"Images/{image_name}", conf=0.25, save=True)

image_path = glob.glob(f"runs/detect/predict*/{image_name}")[0] 

result_image = cv2.imread(image_path)

cv2.imshow('ObjectDetection', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

#print(detection_output)

#print(detection_output[0].numpy())