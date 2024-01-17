# Parking Spot Detection Project
Made as a UNI project
## Overview

This project aims to detect and monitor the occupancy status of parking spots in a given area using the YOLOv8  object detection algorithm. The system identifies cars and boats within predefined parking spaces and logs occupancy changes.

In the project there are 2 models, premade yolo model and one trained on the custom dataset (labeled on roboflow labeler)

## Features

- **Real-time Detection:** Utilizes the YOLOv8 model to perform real-time object detection in a video stream or recorded video.
  
- **Occupancy Monitoring:** Tracks the occupancy status of individual parking spots based on detected vehicles.

- **Logging:** Records occupancy changes in a log file, indicating when a parking spot becomes occupied or vacant.

## Project Structure

- `yolov8s.pt`: YOLOv8 pre-trained model file.
  
- `regions_test.p`: Pickle file containing coordinates of parking spots for region-of-interest.

- `videji/`: Directory containing video files for testing the detection algorithm.

- `parking_log.txt`: Log file recording occupancy changes with timestamps.

- `templates/`: HTML templates for rendering web pages (e.g., index.html).

- `static/`: Static files such as stylesheets or images for the web interface.

- `test_arial_cam.py`: Main Python script running the Flask web application and performing object detection.
