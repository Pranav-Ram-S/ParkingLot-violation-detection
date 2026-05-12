import cv2
import numpy as np

class VehicleDetector:
    """
    Handles preprocessing, background subtraction, and vehicle detection using contours.
    """
    def __init__(self, contour_area_threshold=500, learning_rate=-1):
        # Initialize MOG2 background subtractor
        # detectShadows=True helps in distinguishing shadows from actual objects
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.contour_area_threshold = contour_area_threshold
        self.learning_rate = learning_rate

    def detect(self, frame):
        """
        Process the frame and return bounding boxes and centroids of detected moving objects.
        Returns:
            list of dicts containing 'bbox' (x, y, w, h) and 'centroid' (cx, cy)
        """
        # 1. Preprocessing: Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Background Subtraction to get foreground mask
        fg_mask = self.backSub.apply(blur, learningRate=self.learning_rate)
        
        # 4. Morphological operations to clean up mask (remove noise, fill holes)
        # Threshold the mask to remove shadows (shadows are typically gray, object is white)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 5. Find Contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by area to remove small noise
            if area > self.contour_area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                detections.append({
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'contour': contour
                })
                
        return detections, fg_mask
