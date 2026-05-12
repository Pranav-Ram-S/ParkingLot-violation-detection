import os
import cv2
import glob

def get_frame_source(dataset_path):
    """
    Automatically scans the dataset path.
    If it's a video file, yields frames from the video.
    If it's a directory, yields frames from images in the directory in sorted order.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if os.path.isfile(dataset_path):
        # Treat as video file
        cap = cv2.VideoCapture(dataset_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {dataset_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()
    elif os.path.isdir(dataset_path):
        # Treat as image sequence
        # Supported extensions
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(dataset_path, ext)))
        
        # Sort by filename to maintain temporal sequence
        image_files.sort()
        
        if not image_files:
            raise ValueError(f"No images found in directory: {dataset_path}")
            
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is not None:
                yield frame

def compute_centroid(x, y, w, h):
    """
    Calculate the centroid of a bounding box.
    """
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

def point_in_polygon(point, polygon):
    """
    Checks if a given (x, y) point is inside a polygon.
    polygon is a list of (x, y) tuples.
    """
    # OpenCV's pointPolygonTest is highly optimized for this.
    import numpy as np
    contour = np.array(polygon, dtype=np.int32)
    # Return >= 0 means inside or on edge
    return cv2.pointPolygonTest(contour, point, False) >= 0
