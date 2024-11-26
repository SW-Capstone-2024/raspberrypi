from ultralytics import YOLO
import cv2
import numpy as np
import time
import requests
import subprocess
import re

# model = YOLO("/home/pi/model/yolo8n_6.pt")
# model.export(format="ncnn")
model = YOLO("/home/pi/model/yolo8n_6_ncnn_model")

class_names = ['crack', 'manhole', 'pothole']


class_colors = {
    'crack': (0, 255, 0),       
    'manhole': (0, 0, 255),     
    'pothole': (255, 0, 0),     
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = 1280
frame_height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 20  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
output_filename = '/home/pi/output.mp4'
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

web_server_url = "https://hiroaddetection.run.goorm.site/"

record_duration = 200
start_time = time.time()

def get_location():
    try:
        result = subprocess.run(["adb", "shell", "dumpsys", "location"], capture_output=True, text=True)
        output = result.stdout

        match = re.search(r"last location=Location\[fused (-?\d+\.\d+),(-?\d+\.\d+)", output)
        if match:
            latitude = float(match.group(1))
            longitude = float(match.group(2))
            return latitude, longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_rgb)

        annotated_frame = frame.copy()
        detections = []  
        
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = result.conf[0]
            class_id = int(result.cls[0])
            label = f"{class_names[class_id]}: {conf:.2f}"

            class_name = class_names[class_id]
            color = class_colors.get(class_name, (255, 255, 255)) 

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detection = {
                "class_name": class_name,
                "confidence": float(conf),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections.append(detection)

        latitude, longitude = get_location()

        if detections and latitude is not None and longitude is not None:
            try:
                data = {
                    "detections": detections,
                    "location": {
                        "latitude": latitude,
                        "longitude": longitude
                    }
                }
                response = requests.post(web_server_url, json=data)
                if response.status_code == 200:
                    print("Results and location sent successfully.")
                else:
                    print(f"Failed to send results: {response.status_code}")
            except Exception as e:
                print(f"Error sending results: {e}")

        out.write(annotated_frame)

        if time.time() - start_time > record_duration:
            print("Recording finished.")
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
