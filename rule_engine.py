import time
import math
from utils import point_in_polygon

class RuleEngine:
    """
    Evaluates tracking data against rule sets (e.g., stationary threshold in ROI).
    Tracks simple centroid IDs across frames to measure stationary time.
    """
    def __init__(self, illegal_zones, stationary_threshold_sec, assumed_fps=30.0):
        self.illegal_zones = illegal_zones
        self.stationary_threshold = stationary_threshold_sec
        # We need a tracker to keep vehicle states across frames
        # dict: tracker_id -> {'centroid': (cx, cy), 'first_seen_time': timestamp, 'last_seen_time': timestamp, 'is_violation': bool}
        self.tracked_vehicles = {}
        self.next_id = 1
        
        # Distance threshold to match centroids across frames (in pixels)
        self.max_distance = 50 
        
        # Time increment per frame based on assumed FPS
        self.time_per_frame = 1.0 / assumed_fps
        self.current_time = 0.0

    def _match_detections(self, detections):
        """
        Simple centroid matching across frames (Greedy approach).
        """
        matched_ids = []
        new_tracked = {}
        
        for det in detections:
            cx, cy = det['centroid']
            best_id = None
            min_dist = float('inf')
            
            # Find closest existing tracked vehicle
            for t_id, track_data in self.tracked_vehicles.items():
                if t_id in matched_ids:
                    continue # Already matched
                tcx, tcy = track_data['centroid']
                dist = math.hypot(cx - tcx, cy - tcy)
                
                if dist < self.max_distance and dist < min_dist:
                    min_dist = dist
                    best_id = t_id
            
            if best_id is not None:
                # Update existing track
                new_tracked[best_id] = self.tracked_vehicles[best_id]
                new_tracked[best_id]['centroid'] = (cx, cy)
                new_tracked[best_id]['last_seen_time'] = self.current_time
                matched_ids.append(best_id)
                det['track_id'] = best_id
            else:
                # New vehicle
                new_id = self.next_id
                self.next_id += 1
                new_tracked[new_id] = {
                    'centroid': (cx, cy),
                    'first_seen_time': self.current_time,
                    'last_seen_time': self.current_time,
                    'is_violation': False,
                    'in_zone': False
                }
                det['track_id'] = new_id
                
        # Keep old tracks if they weren't matched but haven't timed out completely (optional)
        # For simplicity in this background subtraction demo, we'll assume if it's not detected, it's gone.
        self.tracked_vehicles = new_tracked

    def process(self, detections):
        """
        Processes frame detections and determines violations.
        Updates each detection dictionary with status information.
        """
        self.current_time += self.time_per_frame
        
        # Update tracker
        self._match_detections(detections)
        
        active_violations = 0
        
        for det in detections:
            track_id = det['track_id']
            cx, cy = det['centroid']
            track_data = self.tracked_vehicles[track_id]
            
            # Check if centroid is inside any illegal zone
            in_zone = False
            for zone in self.illegal_zones:
                if point_in_polygon((cx, cy), zone):
                    in_zone = True
                    break
            
            track_data['in_zone'] = in_zone
            
            # Logic: If it is in the zone, check duration
            # Note: For background subtraction, moving objects are detected. 
            # If a car parks completely, MOG2 will eventually absorb it into background.
            # To track parked cars with MOG2, we need a high history or specialized logic.
            # Here we measure how long the *moving* bounding box has existed inside the zone.
            if in_zone:
                duration_in_zone = self.current_time - track_data['first_seen_time']
                if duration_in_zone >= self.stationary_threshold:
                    track_data['is_violation'] = True
                
                det['duration'] = duration_in_zone
            else:
                # Reset timer if it leaves the zone
                track_data['first_seen_time'] = self.current_time
                track_data['is_violation'] = False
                det['duration'] = 0.0
                
            det['is_violation'] = track_data['is_violation']
            if track_data['is_violation']:
                active_violations += 1
                
        return active_violations
