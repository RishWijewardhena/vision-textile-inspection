"""
Configuration file for stitch measurement system
"""
import cv2
import os
from dotenv import load_dotenv
from hardware_utils import find_esp32 ,find_camera

# Load environment variables from .env file
load_dotenv()

# -------------------------
# Camera Calibration Config
# -------------------------
INTRINSICS_FILE = "camera_calibration.json"
EXTRINSICS_FILE = "extrinsics.json"

# DICT_TYPE = cv2.aruco.DICT_5X5_250
# SQUARES_X = 5 # number of squares in X direction    old setup
# SQUARES_Y = 7 # number of squares in Y direction
# SQUARE_LENGTH = 0.01  # meters (adjust as needed)
# MARKER_LENGTH = 0.007   # meters (adjust as needed)
# MIN_CHARUCO_CORNERS = 6 #as per the openCV documentation

DICT_TYPE = cv2.aruco.DICT_4X4_50
SQUARES_X = 5 # number of squares in X direction
SQUARES_Y = 6 # number of squares in Y direction
SQUARE_LENGTH = 0.010 # meters (adjust as needed)
MARKER_LENGTH = 0.008   # meters (adjust as needed)
MIN_CHARUCO_CORNERS = 6 #as per the openCV documentation
CAPTURE_DELAY = 5  # seconds before auto-capture in extrinsic calibration

# -------------------------
# Camera Settings
# -------------------------
# CAMERA_INDEX = 1

#Get the available camera matrix


CAMERA_INDEX=find_camera()
CALIB_W = 1280
CALIB_H = 960
CAMERA_AUTO_EXPOSURE = 3  # V4L2: 1 = manual, 3 = auto
CAMERA_EXPOSURE = 3.5 # Manual exposure: -10 (darkest) to -4 (brightest). Adjust for lighting conditions.

# -------------------------
# YOLO Model Config
# -------------------------
MODEL_PATH = "best_Model.pt"
STITCH_CLASS_ID = 0   # model class id for stitch
FABRIC_CLASS_ID = 1   # model class id for fabric
CONF_THRESH = 0.20
IOU_THRESH = 0.45 # measures the overlap between two bounding boxes (0 = no overlap, 1 = perfect overlap)
MAX_DETECTIONS = 200

# -------------------------
# Measurement Settings
# -------------------------
FRAME_BUFFER = 8          # median filter across frames
MIN_STITCHES = 3          # minimum stitches to compute average
MAX_EDGE_CANDIDATES = 20  # number of nearest contour points to try per stitch
MAX_PX_DISTANCE = 250    # max pixel distance between stitch centroid and fabric edge (reduced for tighter filtering)
ENVELOPE_NEIGHBORHOOD = 3# columns around centroid to average envelope y
SKIP_CLUSTER = False      # if True, don't try to cluster into 2 stitch lines

# -------------------------
# Serial Communication
# -------------------------
# SERIAL_PORT = "COM4"
#SERIAL_PORT = os.getenv('SERIAL_PORT', '/dev/ttyACM0')
SERIAL_PORT=find_esp32() if find_esp32() is not None else os.getenv('SERIAL_PORT', '/dev/ttyACM0')
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0

# -------------------------
# Database Config  ensuring these are set in .env file or handled gracefully
# -------------------------
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'table': os.getenv('DB_TABLE')
}
# Validate all required configs
required_keys = ['host', 'user', 'password', 'database', 'table']
missing = [key for key, value in DB_CONFIG.items() if value is None]

if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing).upper()}")

# -------------------------
# Application Settings
# -------------------------
INFERENCE_INTERVAL = 2  # seconds between inferences
SAVE_DIR = "saved_annotations"
LOG_DEBUG = True          # set True to print debug info

# -------------------------
# Machine State
# -------------------------
# STATE_IDLE = 'IDLE'
# STATE_RUNNING = 'RUNNING'


# -------------------------
# file cleaner 
# ------------------------
# Delete after 24 hours, check every hour
FILE_RETENTION_HOURS = 24
FILE_CLEANUP_INTERVAL_SECONDS = 3600

# -------------------------
# Activate live imshow windows
# -------------------------
SHOW_WINDOWS = False

# -------------------------
# MQTT Config (Heartbeat)
# -------------------------
MQTT_SERVER = os.getenv("MQTT_SERVER","mqtt.anc.idea8.cloud") #if you cannot get the value from .env, use this default
MQTT_PORT = int(os.getenv("MQTT_PORT","8883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME",'backend')
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD",'bbf12cwcpm')

# device id = DB_TABLE (as you specified)
DEVICE_ID = DB_CONFIG["table"]
MQTT_HEARTBEAT_TOPIC = f"machine/{DEVICE_ID}/status/heartbeat"
MQTT_HEARTBEAT_INTERVAL = 2.0  # seconds
MQTT_TLS_INSECURE="true"