import cv2
import numpy as np
import time
import os
import config
from utils import get_frame_source
from detector import VehicleDetector
from rule_engine import RuleEngine

def draw_zones(frame, zones):
    """Draw illegal parking zones on the frame."""
    for zone in zones:
        pts = np.array(zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=config.ZONE_COLOR, thickness=2)
        # Add a transparent overlay for better visibility
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], config.ZONE_COLOR)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def main():
    print("Starting Illegal Parking Zone Violation Detection System...")
    
    # Initialize components
    try:
        frame_source = get_frame_source(config.DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    detector = VehicleDetector(
        contour_area_threshold=config.CONTOUR_AREA_THRESHOLD,
        learning_rate=config.LEARNING_RATE
    )
    
    rule_engine = RuleEngine(
        illegal_zones=config.ILLEGAL_ZONES,
        stationary_threshold_sec=config.STATIONARY_TIME_THRESHOLD,
        assumed_fps=config.ASSUMED_FPS
    )

    frame_count = 0
    start_time_real = time.time()
    
    cv2.namedWindow('Illegal Parking Detection', cv2.WINDOW_NORMAL)
    
    violation_saved_ids = set() # To prevent saving the same violation multiple times

    for frame in frame_source:
        frame_count += 1
        
        # Make a copy for drawing
        display_frame = frame.copy()
        
        # 1. Detection
        detections, fg_mask = detector.detect(frame)
        
        # 2. Rule Evaluation
        active_violations = rule_engine.process(detections)
        
        # 3. Visualization
        # Draw Zones
        draw_zones(display_frame, config.ILLEGAL_ZONES)
        
        # Draw Detections
        for det in detections:
            x, y, w, h = det['bbox']
            track_id = det['track_id']
            is_viol = det.get('is_violation', False)
            duration = det.get('duration', 0.0)
            
            color = config.BBOX_COLOR_VIOLATION if is_viol else config.BBOX_COLOR_NORMAL
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw centroid
            cx, cy = det['centroid']
            cv2.circle(display_frame, (cx, cy), 4, color, -1)
            
            # Draw text label
            label = f"ID:{track_id}"
            if duration > 0:
                label += f" | {duration:.1f}s"
            if is_viol:
                label += " VIOLATION"
                
            cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 4. Save Screenshot if new violation
            if is_viol and track_id not in violation_saved_ids:
                filename = os.path.join(config.OUTPUT_DIR, f"violation_id{track_id}_frame{frame_count}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"Violation Detected! Saved screenshot: {filename}")
                violation_saved_ids.add(track_id)

        # Calculate and display FPS
        elapsed_real = time.time() - start_time_real
        fps = frame_count / elapsed_real if elapsed_real > 0 else 0
        
        # Top-left info box
        cv2.rectangle(display_frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
        cv2.putText(display_frame, f"Active Violations: {active_violations}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.TEXT_COLOR, 2)
        
        # Display windows
        cv2.imshow('Illegal Parking Detection', display_frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        # Keyboard controls
        key = cv2.waitKey(int(1000/config.ASSUMED_FPS)) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
            
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
