import cv2
from ultralytics import YOLO




from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import pickle
import logging
from datetime import datetime, timedelta

app = Flask(__name__)

# Load your YOLOv8 model
#model = YOLO('yolov8s.pt')
model = YOLO(r'best_homemade.pt')



pickle_file_path = 'regions_parking2.p'

with open(pickle_file_path, 'rb') as file:
    parking_spots = pickle.load(file)

occupancy_status = [False] * len(parking_spots)
prev_occupancy_status = [False] * len(parking_spots)

logging.basicConfig(filename='parking_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

last_change_time = {spot_number: datetime.now() for spot_number in range(0, len(parking_spots) + 1)}
## TO DO : ERROR mozda jer samo an pocetku namjestim vrijeme ili nesto....
MIN_TIME_DIFFERENCE = timedelta(seconds=2)

def log_occupancy_change(spot_number, status):
    global last_change_time

    timestamp = datetime.now()
    time_difference = timestamp - last_change_time[spot_number]

    if time_difference >= MIN_TIME_DIFFERENCE:
        log_message = f"{timestamp} - Parking spot {spot_number} {'occupied' if status else 'free'}"
        log_entry = log_message + '\n'

        with open('parking_log.txt', 'r') as file:
            current_logs = file.read()

        with open('parking_log.txt', 'w') as file:
            file.write(log_entry + current_logs)  # Prepend the new entry

        last_change_time[spot_number] = timestamp



def generate_table():
    table_html = '<table border="1"><tr><th colspan="2">Parkinga mjesta</th></tr>'

    for spot_number, occupied in enumerate(occupancy_status, 1):
        table_html += f'<tr><td>Parking space {spot_number}</td><td style="color: {"red" if occupied else "green"};">{"Occupied" if occupied else "Available"}</td></tr>'

    table_html += '</table>'
    return table_html

def detect_parking():
    global occupancy_status, prev_occupancy_status

 

    video_path = 'videji\parking1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        pred = results[0].boxes.data

        occupancy_status = [False] * len(parking_spots)

        for spot_number, parking_spot in enumerate(parking_spots, 1):
            for det in pred:
                class_id, confidence, x_min, y_min, x_max, y_max = int(det[5]), det[4], det[0], det[1], det[2], det[3]

                if confidence > 0.5 :
                    if x_min > parking_spot[:, 0].min() and x_max < parking_spot[:, 0].max() and \
                            y_min > parking_spot[:, 1].min() and y_max < parking_spot[:, 1].max():
                        if  occupancy_status[spot_number - 1] == False:
                            occupancy_status[spot_number - 1] = True

                        label = f"object (class {class_id}): {confidence:.2f}"
                        color = (0, 255, 0)
                        thickness = 2
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
                        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                    label = f"object (class {class_id}): {confidence:.2f}"
                    color = (0, 255, 0) 
                    thickness = 2
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
                    cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                    #Broj parkinga
                    spot_text = f"Spot {spot_number}"
                    cv2.putText(frame, spot_text, (int(parking_spot[:, 0].mean()), int(parking_spot[:, 1].mean())),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.polylines(frame, [parking_spot], isClosed=True, color=(0, 0, 255), thickness=2)

                
        if occupancy_status != prev_occupancy_status:
            changed_indices = [i for i, (item1, item2) in enumerate(zip(prev_occupancy_status, occupancy_status)) if item1 != item2]
            for index in changed_indices:
                print(f"At parking spot {index}: Value changed from to {occupancy_status[index]}")
                log_occupancy_change(index, occupancy_status[index])

        prev_occupancy_status = list(occupancy_status)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html', table=generate_table())

@app.route('/update_table')
def update_table():
    table_html = generate_table()
    return jsonify(table_html=table_html)

@app.route('/video_feed')
def video_feed():
    return Response(detect_parking(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/read_log')
def read_log():
    with open('parking_log.txt', 'r') as file:
        logs = file.readlines()
    occupancy_change_logs = [log for log in logs if 'Parking spot' in log]

    return jsonify(log_content=occupancy_change_logs)

if __name__ == '__main__':
    app.run(debug=True)

