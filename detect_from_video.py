import cv2
from ultralytics import YOLO


# izaberi koje model  (homemade model ima samo jednu klasu pa će pokazivati samo aute, ali ce pisati da su ljudi)

model = YOLO('yolov8s.pt')
model = YOLO('best_homemade.pt')

# Define class IDs
CAR_CLASS = 2
HUMAN_CLASS = 0
BOAT_CLASS = 8

#cap = cv2.VideoCapture('D:\Projekti\parking spot detection\parking1.mp4')
cap = cv2.VideoCapture(r'D:\Projekti\parking spot detection\videji\ispod_kuce.mp4')



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Perform inference with your YOLOv8 model
    results = model.predict(frame)
    pred = results[0].boxes.data

    for det in pred:
        class_id, confidence, x_min, y_min, x_max, y_max = int(det[5]), det[4], det[0], det[1], det[2], det[3]

        if confidence > 0.5 and (class_id == CAR_CLASS or class_id == HUMAN_CLASS or class_id == BOAT_CLASS):
            if class_id == CAR_CLASS:
                label = f"car: {confidence:.2f}"
                color = (0, 255, 0)  # Green for cars
            elif class_id == BOAT_CLASS:
                label = f"boat: {confidence:.2f}"
                color = (255, 0, 0)  # Blue for boats
            else:
                label = f"person: {confidence:.2f}"
                color = (0, 0, 255)  # Red for people
            
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
