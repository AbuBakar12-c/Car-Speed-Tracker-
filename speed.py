import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from itertools import zip_longest
from tracker import Tracker  # Ensure the correct import if 'Tracker' is from the tracker module
import time
from math import dist

# Load YOLO model
model = YOLO('yolov8s.pt')

# Function to handle RGB mouse callback
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

# Set up OpenCV window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video capture
cap = cv2.VideoCapture('veh2.mp4')

# Read class list from coco.txt
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

count = 0
tracker = Tracker()

# Define the region of interest (cy1, cy2)
cy1 = 322
cy2 = 368
offset = 6

# Initialize dictionaries for vehicle tracking
vh_down = {}
counter = []
vh_up = {}
counter1 = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:  # Skip every 3rd frame
        continue
    frame = cv2.resize(frame, (1020, 500))

    # Make predictions with YOLO model
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Prepare the list of bounding boxes
    bbox_list = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = row
        class_name = class_list[int(class_id)]
        
        # Check if the detected object is a 'car'
        if 'car' in class_name:
            bbox_list.append([int(x1), int(y1), int(x2), int(y2)])

    # Update tracker with bounding boxes
    bbox_id = tracker.update(bbox_list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

        # Vehicle moving down (crossing cy1)
        if cy1 < (cy + offset) and cy1 > (cy - offset):
            vh_down[id] = time.time()

        if id in vh_down:
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                elapsed_time = time.time() - vh_down[id]
                if id not in counter:
                    counter.append(id)
                    distance = 10  # meters
                    speed_ms = distance / elapsed_time
                    speed_kmh = speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{int(speed_kmh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Vehicle moving up (crossing cy2)
        if cy2 < (cy + offset) and cy2 > (cy - offset):
            vh_up[id] = time.time()

        if id in vh_up:
            if cy1 < (cy + offset) and cy1 > (cy - offset):
                elapsed1_time = time.time() - vh_up[id]
                if id not in counter1:
                    counter1.append(id)
                    distance1 = 10  # meters
                    speed_ms1 = distance1 / elapsed1_time
                    speed_kmh1 = speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"{int(speed_kmh1)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow("RGB", frame)

    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
