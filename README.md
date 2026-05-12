# Illegal Parking Zone Violation Detection

This project implements an Image Processing and Computer Vision (IPCV) system to detect illegal parking zone violations. It uses classical computer vision techniques like Background Subtraction and Morphological Operations without relying on heavy Deep Learning models like YOLO.

## Features
- **Background Subtraction**: Uses MOG2 algorithm to separate the foreground (moving vehicles) from the background.
- **Rule Engine**: Allows defining polygonal or rectangular Region of Interest (ROI) illegal parking zones.
- **Stationary Vehicle Tracking**: Tracks vehicle centroids across frames and calculates the time spent inside the restricted zone.
- **Automated Alerts**: Flags a violation if a vehicle is stationary in the restricted zone for more than a configurable threshold (e.g., 10 seconds) and automatically saves a screenshot in the `outputs/` folder.
- **Lightweight & Fast**: Runs entirely on CPU with real-time FPS.

## Project Workflow
1. **Dataset Loading**: Scans the provided directory (e.g., PKLot dataset) and reads images as a sequence to simulate a video feed.
2. **Preprocessing**: Converts frames to grayscale, applies Gaussian Blur to reduce noise.
3. **Background Subtraction**: Uses OpenCV's `createBackgroundSubtractorMOG2` to get a foreground mask.
4. **Morphological Operations**: Cleans the mask using OPEN and CLOSE operations to fill holes and remove small artifacts.
5. **Vehicle Detection**: Finds contours on the cleaned mask and filters them by area to get bounding boxes and centroids for moving vehicles.
6. **ROI Rule Engine**: Tracks centroids across consecutive frames. Checks if they are inside the `ILLEGAL_ZONES` defined in `config.py`.
7. **Stationary Vehicle Analysis**: Calculates the duration a vehicle's centroid remains continuously inside the illegal zone.
8. **Violation Detection**: If the duration exceeds `STATIONARY_TIME_THRESHOLD`, it flags the vehicle as violating and saves a screenshot.

## Folder Structure
```
illegal_parking_detection/
├── main.py             # Entry point. Orchestrates reading, processing, drawing, and saving.
├── detector.py         # Encapsulates preprocessing and MOG2 background subtraction logic.
├── rule_engine.py      # Tracks vehicle centroids and evaluates violation logic.
├── config.py           # Configuration parameters (Thresholds, Paths, Zones).
├── utils.py            # Helper functions for dataset loading, geometry calculations.
├── requirements.txt    # Python dependencies.
├── README.md           # Project documentation.
└── outputs/            # Directory where violation screenshots are automatically saved.
```

## How to Run

1. **Install Dependencies**:
   Ensure you have Python 3.10+ installed. Open a terminal and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Parameters**:
   Open `config.py` and modify settings if needed:
   - `DATASET_PATH`: Ensure it points to your dataset containing images or a video file. (Default points to `C:\Users\prana\Projects\Image processing\Data\train`)
   - `STATIONARY_TIME_THRESHOLD`: Time in seconds before marking a violation.
   - `ILLEGAL_ZONES`: Define your restricted areas using list of coordinates (x, y).

3. **Execute the Application**:
   Run the main script:
   ```bash
   python main.py
   ```

4. **Controls**:
   - The processed video and foreground mask will be shown in live windows.
   - Press the `q` key on your keyboard to quit the application.

## Note on Background Subtraction
Since Background Subtraction (MOG2) detects *moving* objects, a vehicle that enters the frame and comes to a complete halt will eventually blend into the background model if it stays perfectly still for a very long time. For the purpose of this mini-project, we capture the violation *during* the time the vehicle is actively being tracked as it enters and slows down in the zone, or by adjusting the MOG2 `history` and `learningRate` parameters.
