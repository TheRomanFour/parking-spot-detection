import cv2
from ultralytics import YOLO
import numpy as np
import pickle

# Load your YOLOv8 model
model = YOLO('yolov8s.pt')

# Define class IDs
CAR_CLASS = 2

# Load parking spot coordinates from the pickle file
pickle_file_path = 'D:\\Projekti\\parking spot detection\\regions.p'
with open(pickle_file_path, 'rb') as file:
    parking_spots = pickle.load(file)

cap = cv2.VideoCapture(r'D:\Projekti\parking spot detection\videji\parking1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference with your YOLOv8 model
    results = model.predict(frame)
    pred = results[0].boxes.boxes

    # Iterate through each parking spot
    for spot_number, parking_spot in enumerate(parking_spots, 1):
        # Draw a border around the parking spot
        cv2.polylines(frame, [parking_spot], isClosed=True, color=(0, 0, 255), thickness=2)

        # Check for cars in the current parking spot
        for det in pred:
            class_id, confidence, x_min, y_min, x_max, y_max = int(det[5]), det[4], det[0], det[1], det[2], det[3]

            if confidence > 0.5 and class_id == CAR_CLASS:
                if x_min > parking_spot[:, 0].min() and x_max < parking_spot[:, 0].max() and \
                        y_min > parking_spot[:, 1].min() and y_max < parking_spot[:, 1].max():
                    print(f"A car is in parking spot {spot_number}!")

                label = f"car: {confidence:.2f}"
                color = (0, 255, 0)  # Green for cars
                thickness = 2
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
                cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                # Display spot number next to the border
                spot_text = f"Spot {spot_number}"
                cv2.putText(frame, spot_text, (int(parking_spot[:, 0].mean()), int(parking_spot[:, 1].mean())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
