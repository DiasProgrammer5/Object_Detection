import cv2
from ultralytics import YOLO
import random

# Opening the file with the names in read mode
classes_file = open("classes_names.txt", "r")
data = classes_file.read()
class_list = data.split("\n")
classes_file.close()

# Generate random colors for the boxes
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((r,g,b))

model = YOLO("yolov8n.pt", "v8")

# Video capture of the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't read frame")
        break
    
    detect_params = model.predict(source=frame, conf=0.30, save=False)
    
    dp = detect_params[0].numpy()
    
    if len(dp) != 0:
        for i in range(len(detect_params[0])):
            
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[int(clsID)], 3)

            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, class_list[int(clsID)] + " " + str(round(conf,3)) + "%", (int(bb[0]), int(bb[1])-10), font, 1, (255,255,255),2)

    cv2.imshow('ObjectDetection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()