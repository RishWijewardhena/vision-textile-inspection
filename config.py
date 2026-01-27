"""
Configuration file for stitch measurement system
"""
import cv2

# -------------------------
# Camera Calibration Config
# -------------------------
INTRINSICS_FILE = "camera_calibration.json"
EXTRINSICS_FILE = "extrinsics.json"
# DICT_TYPE = cv2.aruco.DICT_5X5_50
DICT_TYPE = cv2.aruco.DICT_4X4_250
SQUARES_X = 5 # number of squares in X direction
SQUARES_Y = 7 # number of squares in Y direction
# SQUARE_LENGTH = 0.012   # meters (adjust as needed)
# MARKER_LENGTH = 0.009   # meters (adjust as needed)
SQUARE_LENGTH = 0.008   # meters (adjust as needed)
MARKER_LENGTH = 0.006   # meters (adjust as needed)
MIN_CHARUCO_CORNERS = 6 #as per the openCV documentation
CAPTURE_DELAY = 10  # seconds before auto-capture in extrinsic calibration

# -------------------------
# Camera Settings
# -------------------------
# CAMERA_INDEX = 1
CAMERA_INDEX="/dev/video0"
CALIB_W = 1280
CALIB_H = 960
CAMERA_EXPOSURE = 5  # Manual exposure: -10 (darkest) to -4 (brightest). Adjust for lighting conditions.

# -------------------------
# YOLO Model Config
# -------------------------
MODEL_PATH = "best_Model.pt"
STITCH_CLASS_ID = 0   # model class id for stitch
FABRIC_CLASS_ID = 1   # model class id for fabric
CONF_THRESH = 0.20
IOU_THRESH = 0.45
MAX_DETECTIONS = 200

# -------------------------
# Measurement Settings
# -------------------------
FRAME_BUFFER = 8          # median filter across frames
MIN_STITCHES = 3          # minimum stitches to compute average
MAX_EDGE_CANDIDATES = 20  # number of nearest contour points to try per stitch
MAX_PX_DISTANCE = 150    # max pixel distance between stitch centroid and fabric edge (reduced for tighter filtering)
ENVELOPE_NEIGHBORHOOD = 3 # columns around centroid to average envelope y
SKIP_CLUSTER = False      # if True, don't try to cluster into 2 stitch lines

# -------------------------
# Serial Communication
# -------------------------
# SERIAL_PORT = "COM4"
SERIAL_PORT="/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0

# -------------------------
# Database Config
# -------------------------
DB_CONFIG = {
    'host': 'aandc.siamsi.net',
    'user': 'root',
    'password': 'P@ssword@a&c',
    'database': 'thread',
    'table': 'machine_test'
}

# -------------------------
# Application Settings
# -------------------------
INFERENCE_INTERVAL = 0.5  # seconds between inferences
SAVE_DIR = "saved_annotations"
LOG_DEBUG = True          # set True to print debug info

# -------------------------
# Machine State
# -------------------------
STATE_IDLE = 'IDLE'
STATE_RUNNING = 'RUNNING'


# -------------------------
# file cleaner 
# ------------------------
# Delete after 24 hours, check every hour
FILE_RETENTION_HOURS = 24
FILE_CLEANUP_INTERVAL_SECONDS = 3600