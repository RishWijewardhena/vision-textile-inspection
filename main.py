"""
Main orchestrator - integrates all modules
"""
import os
import sys
import time
import cv2
import threading
import subprocess
from datetime import datetime
import random

# Import all modules
from config import *
from calibration import run_extrinsic_calibration, create_charuco_board
from serial_reader import SerialReader
from database import DatabaseHandler
from measurement import StitchMeasurementApp ,force_camera_resolution
from file_cleaner import FileCleanerThread
from hardware_utils import find_camera
from backup_data import BackupDataBuffer

from collections import deque

# MQTT heartbeat thread (create this file: mqtt_heartbeat.py)
from mqtt_heartbeat import MqttHeartbeat
from needle_angle_measure import NeedleAngleDetector


def ts():
    """Return current timestamp in format: [HH:MM:SS]"""
    return datetime.now().strftime("[%H:%M:%S]")


class NeedleAngleWorker(threading.Thread):
    def __init__(self, model_path, interval_sec, rotated_angle_threshold):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.interval_sec = interval_sec
        self.rotated_angle_threshold = rotated_angle_threshold
        self._stop_event = threading.Event()
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame = None
        self._busy = False
        self._last_submitted_at = 0.0
        self._latest_result = {
            "rotated": False,
            "detections": [],
            "h_angle": None,
            "v_angle": None,
        }

    def maybe_submit(self, frame, now):
        with self._lock:
            if self._busy or now - self._last_submitted_at < self.interval_sec:
                return False
            self._pending_frame = frame.copy()
            self._busy = True
            self._last_submitted_at = now
            self._frame_ready.set()
            return True

    def latest_result(self):
        with self._lock:
            return dict(self._latest_result)

    def run(self):
        detector = None
        while not self._stop_event.is_set():
            if not self._frame_ready.wait(timeout=0.5):
                continue

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._frame_ready.clear()

            if frame is None:
                with self._lock:
                    self._busy = False
                continue

            try:
                if detector is None:
                    detector = NeedleAngleDetector(
                        model_path=self.model_path,
                        rotated_angle_threshold=self.rotated_angle_threshold,
                    )

                result = detector.measure_frame(frame, annotate=False)
                result["checked_at"] = time.time()
                h_angle = result.get("h_angle")
                h_text = f"{h_angle:.1f}" if h_angle is not None else "N/A"
                rotated = result.get("rotated", False)
                print(ts() + f" 🧭 Needle angle checked: H={h_text}°, rotated={rotated}")

            except Exception as exc:
                result = {
                    "rotated": False,
                    "detections": [],
                    "h_angle": None,
                    "v_angle": None,
                    "error": str(exc),
                    "checked_at": time.time(),
                }
                print(ts() + f" ⚠️ Needle angle inference failed: {exc}")

            with self._lock:
                self._latest_result = result
                self._busy = False

    def stop(self):
        self._stop_event.set()
        self._frame_ready.set()


def run_startup_calibration():
    """Run extrinsic calibration at startup  """
    print(ts() + " " + "\n" + "="*60)
    print(ts() + " SYSTEM STARTUP - EXTRINSIC CALIBRATION")
    print(ts() + " " + "="*60)
    
    board, detector = create_charuco_board()
    success = run_extrinsic_calibration(board, detector)

    if success:
        print(ts() + " ✅ CALIBRATION COMPLETE")
        cv2.destroyAllWindows()  # Ensure calibration window is closed
        cv2.waitKey(1)
        return True

    # Fallback: keep working with existing extrinsics if available
    elif not success and os.path.exists(EXTRINSICS_FILE):
        print(ts() + " ⚠️ Calibration failed, using existing extrinsics file:", EXTRINSICS_FILE)
        cv2.destroyAllWindows()  # Ensure calibration window is closed
        cv2.waitKey(1)
        return True
    
    else:
        print(ts() + " ❌ CALIBRATION FAILED - Cannot continue without extrinsics")
        print(ts() + " Please ensure:")
        print(ts() + "   1. ChArUco board is visible to camera")
        print(ts() + "   2. Lighting is adequate")
        print(ts() + "   3. Board is on the measurement plane")
        return False


def reload_camera():
    """Reload webcam driver (uvcvideo)."""
    print(ts() + " 🔄 Reloading webcam driver...")
    try:
        subprocess.run(["sudo", "modprobe", "-r", "uvcvideo"], check=True)
        subprocess.run(["sudo", "modprobe", "uvcvideo"], check=True)
        print(ts() + " ✅ Webcam driver reloaded")
    except subprocess.CalledProcessError as e:
        print(ts() + f" ⚠️ Failed to reload webcam driver: {e}")


def main():
    """Main application loop"""
    print(ts() + " " + "\n" + "="*60)
    print(ts() + " 🧵 STITCH MEASUREMENT SYSTEM")
    print(ts() + " " + "="*60)
    
    # commenting out the extrinsics calibration for speed, but you can uncomment to run calibration at startup. Just make sure to have the ChArUco board in view and good lighting for it to work.

    # # Step 1: Run calibration
    # try:
    #     calibrated = run_startup_calibration()
    #     if not calibrated:
    #         sys.exit(1) # Exit if calibration failed

    # except Exception as e:
        
    #     print(f"\n❌ Calibration error: {e}")
    #     sys.exit(1)
    
    # time.sleep(1)  # Brief pause after calibration


    
    # Step 2: Initialize all components
    print(ts() + " \n📡 Initializing components...")
    
    try:
        # Initialize measurement app
        measurement_app = StitchMeasurementApp(
            calib_path=INTRINSICS_FILE,
            extr_path=EXTRINSICS_FILE,
            model_path=MODEL_PATH,
            camera_index=CAMERA_INDEX,
            calib_w=CALIB_W,
            calib_h=CALIB_H,
            frame_buffer=FRAME_BUFFER,
            min_stitches=MIN_STITCHES,
            stitch_id=STITCH_CLASS_ID,
            fabric_id=FABRIC_CLASS_ID
        )
        print(ts() + " ✅ Measurement app initialized")
        
    except Exception as e:
        print(ts() + f" ❌ Failed to initialize measurement app: {e}")
        sys.exit(1)
    
    # Initialize database
    db = DatabaseHandler()
    db_connected = db.connect()
    if not db_connected:
        print(ts() + " ⚠️ Database connection failed at startup - will retry on next measurement")
    
    # Initialize backup data buffer (for failed measurements)
    backup_buffer = BackupDataBuffer()
    
    # Note: db object is kept even if connection fails, so reconnection can be attempted
    last_date=db.get_last_record_date()
    today=datetime.now().date()
    if last_date and last_date!=today:
        db.insert_measurement(
            total_distance=0.0,
            stitch_length=0.0,
            seam_allowance=0.0,
        )
        print(ts() + " 🔄 New day detected - total distance reset to 0 in database")

    elif last_date is None:
        db.insert_measurement(
            total_distance=0.0,
            stitch_length=0.0,
            seam_allowance=0.0,
        )
        print(ts() + " 📊 No previous records - total distance initialized to 0 in database")
        
    else:
        print(ts() + f" 📊 Total distance continues from last measurement in database: {last_date}")

    
    # Initialize serial reader
    serial_reader = SerialReader()
    if not serial_reader.start_reading():
        print(ts() + " ⚠️ Serial connection failed - continuing without serial data")
        serial_reader = None
    
    #initialize file cleaner
    file_cleaner=FileCleanerThread()
    file_cleaner.start()

    # Initialize MQTT heartbeat
    heartbeat = None
    reset_requested = threading.Event()

    def queue_reset_request():
        """Queue reset work to run inside the main loop thread."""
        reset_requested.set()

    try:
        # Use MQTT constants 
        heartbeat = MqttHeartbeat(
            broker=MQTT_SERVER,
            port=MQTT_PORT,
            username=MQTT_USERNAME,
            password=MQTT_PASSWORD,
            topic=MQTT_HEARTBEAT_TOPIC,
            interval_sec=MQTT_HEARTBEAT_INTERVAL,
            tls_insecure=MQTT_TLS_INSECURE,
            reset_topic=MQTT_RESET_TOPIC,
            on_reset=queue_reset_request,
            esp32_issue_topic=MQTT_ESP32_ISSUE_TOPIC,
        )
        heartbeat.start()
        print(ts() +
            f" ✅ MQTT heartbeat started: {MQTT_HEARTBEAT_TOPIC} "
            f"(every {MQTT_HEARTBEAT_INTERVAL}s), reset listener: {MQTT_RESET_TOPIC}"
        )
    except Exception as e:
        print(ts() + f" ⚠️ MQTT heartbeat not started: {e} (continuing without heartbeat)")

    angle_worker = None
    try:
        angle_worker = NeedleAngleWorker(
            model_path=NEEDLE_ANGLE_MODEL_PATH,
            interval_sec=NEEDLE_ANGLE_CHECK_INTERVAL,
            rotated_angle_threshold=NEEDLE_ROTATED_ANGLE_THRESHOLD,
        )
        angle_worker.start()
        print(
            ts()
            + f" ✅ Needle angle worker started: {NEEDLE_ANGLE_MODEL_PATH} "
            + f"(every {NEEDLE_ANGLE_CHECK_INTERVAL}s, threshold {NEEDLE_ROTATED_ANGLE_THRESHOLD}°)"
        )
    except Exception as e:
        print(ts() + f" ⚠️ Needle angle worker not started: {e}")



    print(ts() + " " + "\n" + "="*60)
    print(ts() + " 🎯 SYSTEM READY - Starting measurements")
    print(ts() + " " + "="*60)
    print(ts() + " Press 'q' to quit")
    print(ts() + " " + "="*60 + "\n")
    
    # Step 3: Main measurement loop
    RESET_POST_DELAY_SEC = 2.0
    last_inference_time = 0
    frame_count = 0
    last_stitch_count = 0

    # step 3.1: Getting last total distance from DB to continue from previous session if available
    total_distance_mm = float(db.get_last_record_total_distance() if db else 0.0) 
    if LOG_DEBUG:
        print(ts() + f" 📊 Starting total distance: {total_distance_mm:.2f}mm")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create session-specific folder for this run
    session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(SAVE_DIR, session_start)
    os.makedirs(session_dir, exist_ok=True)
    print(ts() + f" 📁 Session folder: {os.path.abspath(session_dir)}")

    CAMERA_RECONNECT_ATTEMPTS = 0
    MAX_RECONNECT_ATTEMPTS = 10

    # Initialize variables to prevent UnboundLocalError
    stitch_delta = 0
    moved_distance_mm = 0.0
    
    # ESP32 connection tracking
    esp32_issue_published = False  # Track if we've already published ESP32 issue
    last_esp32_issue_publish_time = 0  # Track last publish time
    ESP32_ISSUE_PUBLISH_INTERVAL = 2.0  # Publish every 2 seconds while disconnected

    # Raw-history buffers (post-offset) used to detect sustained changes
    raw_seam_history = deque(maxlen=20)
    raw_width_history = deque(maxlen=20)

    # Buffer for last 5 valid measurements
    valid_seam_buffer = deque(maxlen=5)
    valid_width_buffer = deque(maxlen=5)


    #retrieve last 5 records from DB to pre-fill smoothing buffers and continue from previous session trends if available
    def initialize_buffers_from_db():
        if db:
            last_records = db.get_last_n_records(5)
            print(ts() + f" 📊 Retrieved last {len(last_records)} records from DB \n {last_records}")
            for record in last_records:
                if record['seam_allowance'] is not None :
                    valid_seam_buffer.append(float(record['seam_allowance']))
                if record['stitch_length'] is not None :
                    valid_width_buffer.append(float(record['stitch_length']))
            print(ts() + f" 📊 Pre-filled smoothing buffers with last {len(valid_seam_buffer)} seam and {len(valid_width_buffer)} width measurements from DB")
        

        print(ts() + f" 📊 Initial valid seam buffer: {list(valid_seam_buffer)}")
        print(ts() + f" 📊 Initial valid width buffer: {list(valid_width_buffer)}")

    #initialize smoothing buffers with recent DB values to allow smoother startup if historical data exists
    initialize_buffers_from_db()

    # reset the total distance in the database to 0 at startup
    if serial_reader:
        serial_success = serial_reader.send_command("R")


    def perform_reset():
        """Reset DB values, ESP32 count, and runtime smoothing state."""
        nonlocal total_distance_mm, last_stitch_count, stitch_delta, moved_distance_mm, esp32_issue_published, last_esp32_issue_publish_time

        stitch_delta = 0
        moved_distance_mm = 0.0
        esp32_issue_published = False  # Reset the ESP32 issue flag
        last_esp32_issue_publish_time = 0  # Reset publish time
        print(ts() + " 🔁 Processing reset command...")

        db_success = db.insert_measurement(
            total_distance=0.0,
            stitch_length=0.0,
            seam_allowance=0.0,
        )
        if db_success:
            print(ts() + " ✅ Reset DB insert succeeded (all zeros)")
            # Try to flush backup buffer after successful reset
            if not backup_buffer.is_empty():
                if backup_buffer.flush_to_db(db):
                    print(ts() + f" ✅ Also flushed {len(backup_buffer.get_all())} pending measurements")
        else:
            print(ts() + " ⚠️ Reset DB insert failed (will retry on next measurement)")
            # Add reset record to backup if DB fails
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            backup_buffer.add(timestamp, 0.0, 0.0, 0.0)

        serial_success = False
        if serial_reader:
            serial_success = serial_reader.send_command("R")
            if serial_success:
                print(ts() + " ✅ Serial reset command sent: R")
            else:
                print(ts() + " ⚠️ Serial reset command failed")
        else:
            print(ts() + " ⚠️ Serial reader unavailable for reset command")

        # Give ESP32 time to apply reset before using stitch count baseline again.
        time.sleep(RESET_POST_DELAY_SEC)

        total_distance_mm = 0.0
        last_stitch_count = serial_reader.get_stitch_count() if serial_reader else 0
        valid_seam_buffer.clear()
        valid_width_buffer.clear()
        initialize_buffers_from_db() # re-populate buffers from DB after reset to allow smoother recovery if historical data exists
        print(ts() + " ✅ Runtime counters and buffers reset")

        if db_success and serial_success and heartbeat:
            heartbeat.publish_reset_success()
            print(ts() + f" ✅ MQTT reset acknowledgment published: {MQTT_RESET_TOPIC} -> reset_success")

    
    try:
        while True:
            if reset_requested.is_set():
                reset_requested.clear()
                perform_reset()
            
            # ===== CHECK DATABASE RECONNECTION =====
            # Check if DB reconnected since last failure
            if not db.connection or not db.connection.is_connected():
                if db.connect():
                    print(ts() + " 🔄 Database reconnected - flushing backup buffer")
                    if not backup_buffer.is_empty():
                        if backup_buffer.flush_to_db(db):
                            print(ts() + f" ✅ Flushed {len(backup_buffer.get_all())} buffered measurements")
            
            # ===== CHECK ESP32 CONNECTION =====
            if serial_reader:
                current_loop_time = time.time()
                if serial_reader.is_connected():
                    # ESP32 is connected - clear the issue flag
                    esp32_issue_published = False
                    last_esp32_issue_publish_time = 0
                else:
                    # ESP32 is NOT connected - publish issue repeatedly every N seconds
                    if current_loop_time - last_esp32_issue_publish_time >= ESP32_ISSUE_PUBLISH_INTERVAL:
                        print(ts() + " ⚠️ ESP32 disconnected or not responding")
                        if heartbeat:
                            try:
                                heartbeat.client.publish(
                                    MQTT_ESP32_ISSUE_TOPIC,
                                    payload="issue",
                                    qos=0,
                                    retain=False,
                                )
                                print(ts() + f" 📡 MQTT ESP32 issue sent: {MQTT_ESP32_ISSUE_TOPIC} -> issue")
                                last_esp32_issue_publish_time = current_loop_time
                                esp32_issue_published = True
                            except Exception as exc:
                                print(ts() + f" ⚠️ MQTT ESP32 issue publish failed: {exc}")
            
            # get frame from camera
            ret, frame = measurement_app.cap.read()
            
            if not ret:
                CAMERA_RECONNECT_ATTEMPTS += 1
                print(ts() + f" ⚠️ No frame from camera (attempt {CAMERA_RECONNECT_ATTEMPTS}/{MAX_RECONNECT_ATTEMPTS})")
                
                # If we have a heartbeat MQTT client, publish an issue message on each failed attempt to alert the system
                if heartbeat:
                    try:
                        heartbeat.client.publish(
                                MQTT_CAMERA_ISSUE_TOPIC,
                                payload="issue",
                                qos=0,
                                retain=False,
                            )
                        print(ts() + f" 📡 MQTT camera issue sent: {MQTT_CAMERA_ISSUE_TOPIC} -> issue")
                    except Exception as exc:
                        print(ts() + f" ⚠️ MQTT camera issue publish failed: {exc}")

                if CAMERA_RECONNECT_ATTEMPTS >= MAX_RECONNECT_ATTEMPTS:
                    print(ts() + " ❌ Camera disconnected. Reloading usb_storage and attempting reconnect...")

                    measurement_app.cap.release()
                    time.sleep(2)
                    reload_camera()  # reload the webcam driver
                    time.sleep(5)    # wait for driver to stabilize

                    new_camera_index = find_camera()
                    
                    # Verify VideoCapture opened successfully
                    new_cap = cv2.VideoCapture(new_camera_index, cv2.CAP_V4L2)
                    
                    # Force MJPG compression FIRST (before resolution settings for proper negotiation)
                    new_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    if not new_cap.isOpened():
                        print(ts() + f" ⚠️ Failed to open camera at {new_camera_index}")
                        time.sleep(1)
                        continue  # Skip this iteration and retry
                    
                    measurement_app.cap = new_cap
                    
                    # Validate resolution settings
                    aw, ah = force_camera_resolution(measurement_app.cap, CALIB_W, CALIB_H)
                    if aw == 0 or ah == 0:
                        print(ts() + f" ❌ Camera resolution failed (got {aw}x{ah}). Retrying...")
                        time.sleep(1)
                        continue  # Skip and retry
                    
                    if aw != CALIB_W or ah != CALIB_H:
                        print(ts() + f" ⚠️ Resolution mismatch: got {aw}x{ah}, expected {CALIB_W}x{CALIB_H}")
                    
                    # Test camera produces frames
                    test_success = False
                    for test_attempt in range(3):
                        time.sleep(0.5)
                        test_ret, test_frame = measurement_app.cap.read()
                        if test_ret and test_frame is not None:
                            test_success = True
                            print(ts() + f" ✅ Camera test frame successful")
                            break
                        else:
                            print(ts() + f" ⚠️ Camera test frame failed (attempt {test_attempt+1}/3)")
                    
                    if not test_success:
                        print(ts() + f" ❌ Camera not producing frames after reconnect")
                        time.sleep(1)
                        continue  # Skip and retry
                    
                    CAMERA_RECONNECT_ATTEMPTS = 0
                    print(ts() + f" ✅ Camera successfully reconnected: {new_camera_index}")
                    
                continue

            CAMERA_RECONNECT_ATTEMPTS = 0  # Reset on successful frame
            current_time = time.time()

            
            # Process frame at intervals
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                # Get measurements from vision system
                annotated, measurements = measurement_app.process_frame(frame)

                if angle_worker and angle_worker.maybe_submit(frame, current_time):
                    print(ts() + " 🧭 Needle angle inference queued")

                if angle_worker and heartbeat:
                    angle_result = angle_worker.latest_result()
                    if angle_result.get("rotated"):
                        try:
                            heartbeat.client.publish(
                                MQTT_CAMERA_ISSUE_TOPIC,
                                payload="rotated",
                                qos=0,
                                retain=False,
                            )
                            print(ts() + f" 📡 MQTT camera issue sent: {MQTT_CAMERA_ISSUE_TOPIC} -> rotated")
                        except Exception as exc:
                            print(ts() + f" ⚠️ MQTT camera rotated publish failed: {exc}")
                
                # Get stitch count from serial
                current_stitch_count = serial_reader.get_stitch_count() if serial_reader else last_stitch_count

                if last_stitch_count > current_stitch_count:
                    last_stitch_count = current_stitch_count
                    print(ts() + " 🔄  Stitch count reset detected - updating baseline to current count:", last_stitch_count)


                # Calculate movement based on stitch count change
                stitch_delta += current_stitch_count - last_stitch_count
                last_stitch_count = current_stitch_count

                # measurements is a dict with keys: edge_distance_mm, stitch_width_mm, stitch_count, timestamp
                seam_length_mm = measurements.get('edge_distance_mm', None)  # top_distance
                stitch_width_mm = measurements.get('stitch_width_mm', None)

                if seam_length_mm is not None:
                    seam_length_mm += SEAM_LENGTH_OFFSET
                    cv2.putText(annotated, f"Adjusted seam: {seam_length_mm:.2f}mm", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if stitch_width_mm is not None:
                    stitch_width_mm += STITCH_WIDTH_OFFSET
                    cv2.putText(annotated, f"Adjusted width: {stitch_width_mm:.2f}mm", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # store offset-applied raw values for history checks
                if seam_length_mm is not None:
                    raw_seam_history.append(seam_length_mm)
                if stitch_width_mm is not None:
                    raw_width_history.append(stitch_width_mm)

                if LOG_DEBUG:
                    raw_seam = measurements.get("edge_distance_mm")
                    raw_width = measurements.get("stitch_width_mm")

                    # print(
                    #     ts() + 
                    #     f" 🔍 Raw measurements: "
                    #     f"seam={f'{raw_seam:.2f}' if raw_seam is not None else 'N/A'}mm, "
                    #     f"width={f'{raw_width:.2f}' if raw_width is not None else 'N/A'}mm"
                    # )

                # Determine if this is a valid measurement
                valid_seam = (
                    seam_length_mm is not None
                    and Seam_lower_limit < seam_length_mm < Seam_upper_limit
                )

                valid_stitch = (
                    stitch_width_mm is not None
                    and stitch_lower_limit < stitch_width_mm < stitch_upper_limit
                )

                confirmed_override = False

                # If seam is above soft upper limit, check for N consecutive similar samples -> accept
                if not valid_seam and seam_length_mm is not None and seam_length_mm > Seam_upper_limit:
                    recent = [v for v in list(raw_seam_history)[-CONFIRM_CONSECUTIVE:] if v is not None]
                    if len(recent) >= CONFIRM_CONSECUTIVE and all(v > Seam_upper_limit - CONFIRM_TOLERANCE_MM for v in recent):
                        valid_seam = True
                        confirmed_override = True
                        print(ts() + f" 🚨Seam length above {Seam_upper_limit}mm but sustained for {CONFIRM_CONSECUTIVE} samples - accepting as valid")
                if not valid_seam and seam_length_mm is not None and seam_length_mm < Seam_lower_limit:
                    recent = [v for v in list(raw_seam_history)[-CONFIRM_CONSECUTIVE:] if v is not None]
                    if len(recent) >= CONFIRM_CONSECUTIVE and all(v < Seam_lower_limit + CONFIRM_TOLERANCE_MM for v in recent):
                        valid_seam = True
                        confirmed_override = True
                        print(ts() + f" 🚨Seam length below {Seam_lower_limit}mm but sustained for {CONFIRM_CONSECUTIVE} samples - accepting as valid"
                        )
                # For small/too-low measurements: ignore (do not confirm below lower bound)
                # If stitch width is above soft upper limit, check similarly
                if not valid_stitch and stitch_width_mm is not None and stitch_width_mm > stitch_upper_limit:
                    recent_w = [v for v in list(raw_width_history)[-CONFIRM_CONSECUTIVE:] if v is not None]
                    if len(recent_w) >= CONFIRM_CONSECUTIVE and all(v > stitch_upper_limit - CONFIRM_TOLERANCE_MM for v in recent_w):
                        valid_stitch = True
                        confirmed_override = True
                        print(ts() + f" 🚨Stitch width above {stitch_upper_limit}mm but sustained for {CONFIRM_CONSECUTIVE} samples - accepting as valid")
                if not valid_stitch and stitch_width_mm is not None and stitch_width_mm < stitch_lower_limit:
                    recent_w = [v for v in list(raw_width_history)[-CONFIRM_CONSECUTIVE:] if v is not None]
                    if len(recent_w) >= CONFIRM_CONSECUTIVE and all(v < stitch_lower_limit + CONFIRM_TOLERANCE_MM for v in recent_w):
                        valid_stitch = True
                        confirmed_override = True
                        print(ts() + f" 🚨Stitch width below {stitch_lower_limit}mm but sustained for {CONFIRM_CONSECUTIVE} samples - accepting as valid")


                has_valid_measurement = valid_seam and valid_stitch                
                # If valid, save to buffer save to smoothing buffers (if a confirmed override happened, adapt faster)
                if has_valid_measurement:
                    if confirmed_override:
                        pass
                        # valid_seam_buffer.clear()
                        # valid_width_buffer.clear()
                    valid_seam_buffer.append(seam_length_mm)
                    valid_width_buffer.append(stitch_width_mm)
                    if LOG_DEBUG:
                        print(ts() + f" 📦 Buffered measurement: seam={seam_length_mm:.2f}mm, width={stitch_width_mm:.2f}mm "
                            f"(buffer size: {len(valid_seam_buffer)}/5)")
                else:
                    # fallback: use buffered average if available 
                    if len(valid_seam_buffer) > 0 and len(valid_width_buffer) > 0:
                        seam_length_mm = sum(valid_seam_buffer) / len(valid_seam_buffer)
                        stitch_width_mm = sum(valid_width_buffer) / len(valid_width_buffer)
                        has_valid_measurement = True
                        if LOG_DEBUG:
                            print(ts() + f"  Using buffered average: seam={seam_length_mm:.2f}mm, "
                                f"width={stitch_width_mm:.2f}mm (from {len(valid_seam_buffer)} samples)")
                if stitch_delta > 0 and has_valid_measurement:
                    # Calculate moved distance
                    moved_distance_mm = stitch_delta * stitch_width_mm
                    total_distance_mm += moved_distance_mm
    
                    # Insert to database
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    success = db.insert_measurement(
                        total_distance=round(total_distance_mm, 1),
                        stitch_length=round(stitch_width_mm, 1),
                        seam_allowance=round(seam_length_mm, 1)
                    )
                    
                    if not success:
                        # Add to backup buffer instead of dropping data
                        backup_buffer.add(timestamp, round(total_distance_mm, 1), 
                                        round(stitch_width_mm, 1), round(seam_length_mm, 1))
                        print(ts() + f" ⚠️ Database insert failed - backed up to buffer ({backup_buffer.size()}/50)")
                    else:
                        # On successful insert, try to flush any pending backups
                        if not backup_buffer.is_empty():
                            print(ts() + f" 📊 DB insert successful. Buffer has {backup_buffer.size()} pending items")
                    
                    # Update total distance
                    seam_display = f"{seam_length_mm:.2f}" if seam_length_mm is not None else "N/A"
                    info_text = (f"Count: {current_stitch_count} | Count_delta: {stitch_delta} | Moved: {moved_distance_mm:.2f}mm | "
                               f"Total: {total_distance_mm:.2f}mm | Seam: {seam_display}mm")
                    if stitch_width_mm is not None:
                        info_text += f" | Width: {stitch_width_mm:.2f}mm"
                    
                    cv2.putText(annotated, info_text, (10, annotated.shape[0] - 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    stitch_delta = 0 # reset stitch delta after applying movement
                    print(ts() + f" 📏 {info_text}")
                else:
                    # No valid measurements
                    cv2.putText(annotated, f"Stitch count: {current_stitch_count} (waiting for measurements)", 
                              (10, annotated.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # Save annotated image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(session_dir, f"frame_{frame_count:05d}_{timestamp}.jpg")
                cv2.imwrite(save_path, annotated)
                
                if SHOW_WINDOWS:
                    cv2.imshow("Stitch Measurement System", annotated)
                last_inference_time = current_time
                frame_count += 1
            else:
                # Display live feed without processing
                if SHOW_WINDOWS:
                    cv2.imshow("Stitch Measurement System", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(ts() + " \n🛑 Shutdown requested by user")
                break
    
    except KeyboardInterrupt:
        print(ts() + " \n🛑 Interrupted by user")
    
    finally:
        # Step 4: Cleanup
        print(ts() + " \n🧹 Cleaning up...")
        
        if serial_reader:
            serial_reader.stop()
        
        if db:
            # Try to flush any remaining backup data before closing
            if not backup_buffer.is_empty():
                print(ts() + f" 📊 Attempting to flush {backup_buffer.size()} remaining measurements before shutdown...")
                if backup_buffer.flush_to_db(db):
                    print(ts() + " ✅ Successfully flushed remaining measurements")
                else:
                    print(ts() + f" ⚠️ {backup_buffer.size()} measurements remain in backup (will flush on restart)")
            db.close()

        file_cleaner.stop() #stop file cleaner thread
        
        measurement_app.cap.release()

        if angle_worker:
            angle_worker.stop()
            angle_worker.join(timeout=5)

        if heartbeat:
            heartbeat.stop()
            
        cv2.destroyAllWindows()
        
        print(ts() + f" \n✅ Total frames processed: {frame_count}")
        print(ts() + f" 📁 Images saved to: {os.path.abspath(SAVE_DIR)}")
        print(ts() + " \n👋 System shutdown complete")


if __name__ == "__main__":
    main()