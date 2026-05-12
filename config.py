import os

# Dataset Path
# Provide the path to the directory containing video files OR image sequences.
# The system will auto-scan this folder.
DATASET_PATH = r"C:\Users\prana\Projects\Image processing\Data\train"

# Output directory for saving violation screenshots
OUTPUT_DIR = "outputs"

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing & Detection Parameters
CONTOUR_AREA_THRESHOLD = 500  # Minimum area of foreground contour to be considered a vehicle
LEARNING_RATE = -1  # MOG2 learning rate. -1 means auto.

# Rule Engine Parameters
STATIONARY_TIME_THRESHOLD = 10.0  # seconds. If vehicle stays >= this, it's a violation

# Region of Interest (ROI) for Illegal Parking Zones.
# List of polygons. Each polygon is a list of (x, y) coordinates.
# For example, let's define a sample rectangular zone and a polygonal zone.
ILLEGAL_ZONES = [
    [(300, 200), (800, 200), (800, 400), (300, 400)], # Sample Zone 1
    [(100, 500), (400, 500), (350, 650), (50, 650)]   # Sample Zone 2
]

# Assuming camera FPS if using an image sequence (since images might not have inherent FPS)
# Adjust this based on how fast the sequence was captured. For real-time, 30 FPS is common.
ASSUMED_FPS = 30.0

# Visualization config
ZONE_COLOR = (0, 0, 255) # Red for illegal zone boundaries
TEXT_COLOR = (255, 255, 255)
BBOX_COLOR_NORMAL = (0, 255, 0) # Green for normal moving vehicle
BBOX_COLOR_VIOLATION = (0, 0, 255) # Red for violation
