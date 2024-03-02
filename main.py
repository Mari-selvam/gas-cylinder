from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import serial

ser = serial.Serial('/dev/cu.usbmodem14101',9600)
print(ser.name)

# Replace the path with the path to your video file
# r = r"D:\Project\Gas\How LPG Cylinders are filled in India. HPCL, IOCL, BPCL.mp4"

# cap = cv2.VideoCapture('http://192.0.0.4:8080/video')

cap = cv2.VideoCapture(0)
cap.set(0, 480)
cap.set(0, 640)

model = YOLO(r"best.pt")

classNames = ["Cap_Close","Cap_Open"]

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or unable to read video.")
        break

    results = model(img, stream=True)

    detections = np.empty((0, 6))  # Increase size to accommodate the class name

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            
            print("cls:", cls)  # Debugging output
            
            if 0 <= cls < len(classNames):
                currentClass = classNames[cls]
                
                
                
                # Adjust class names to match your list
                if currentClass in ["Cap_Close","Cap_Open"] and conf > 0.3:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                    
                    if currentClass  == "Cap_Open" and conf > 0.5:
                        ser.write(b"H")
                    # Crop and save the number plate image
                    roi = img[y1:y2, x1:x2]
                    cv2.imwrite("number_plate_image.jpg", roi)
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=2,offset=3)
                    # Add class name to detection array
                    currentArray = np.array([x1, y2, x2, y2, conf, currentClass])
                    detections = np.vstack((detections, currentArray))

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close() 

