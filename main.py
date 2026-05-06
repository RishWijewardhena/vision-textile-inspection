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

from collections import deque

# MQTT heartbeat thread (create this file: mqtt_heartbeat.py)
from mqtt_heartbeat import MqttHeartbeat


def ts():
    """Return current timestamp in format: [HH:MM:SS]"""
    return datetime.now().strftime("[%H:%M:%S]")


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
    if not db.connect():
        print(ts() + " ❌ Database connection failed - continuing without DB")
        db = None




    if db:
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
        # Use MQTT constants from config.py if you added them,
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
        )
        heartbeat.start()
        print(ts() +
            f" ✅ MQTT heartbeat started: {MQTT_HEARTBEAT_TOPIC} "
            f"(every {MQTT_HEARTBEAT_INTERVAL}s), reset listener: {MQTT_RESET_TOPIC}"
        )
    except Exception as e:
        print(ts() + f" ⚠️ MQTT heartbeat not started: {e} (continuing without heartbeat)")



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


    # Raw-history buffers (post-offset) used to detect sustained changes
    raw_seam_history = deque(maxlen=10)
    raw_width_history = deque(maxlen=10)

    # Buffer for last 5 valid measurements
    valid_seam_buffer = deque(maxlen=5)
    valid_width_buffer = deque(maxlen=5)

    # reset the total distance in the database to 0 at startup
    if serial_reader:
        serial_success = serial_reader.send_command("R")



    def perform_reset():
        """Reset DB values, ESP32 count, and runtime smoothing state."""
        nonlocal total_distance_mm, last_stitch_count,stitch_delta, moved_distance_mm

        stitch_delta = 0
        moved_distance_mm = 0.0
        print(ts() + " 🔁 Processing reset command...")

        db_success = False
        if db:
            db_success = db.insert_measurement(
                total_distance=0.0,
                stitch_length=0.0,
                seam_allowance=0.0,
            )
            if db_success:
                print(ts() + " ✅ DB reset row inserted (0,0,0)")
            else:
                print(ts() + " ⚠️ DB reset row insert failed")
        else:
            print(ts() + " ⚠️ DB unavailable for reset row insert")

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
        valid_seam_buffer.extend([6.5] * 5)
        valid_width_buffer.extend([3.9] * 5)
        print(ts() + " ✅ Runtime counters and buffers reset")

        if db_success and serial_success and heartbeat:
            heartbeat.publish_reset_success()
            print(ts() + f" ✅ MQTT reset acknowledgment published: {MQTT_RESET_TOPIC} -> reset_success")

    
    try:
        while True:
            if reset_requested.is_set():
                reset_requested.clear()
                perform_reset()

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

                    reload_camera()

                    measurement_app.cap.release()
                    time.sleep(2)

                    new_camera_index = find_camera()

                    measurement_app.cap = cv2.VideoCapture(new_camera_index, cv2.CAP_V4L2)
                    force_camera_resolution(measurement_app.cap, CALIB_W, CALIB_H)
                    CAMERA_RECONNECT_ATTEMPTS = 0
                    print(ts() + f" 🔁 Re-detected camera: {new_camera_index}")
                    time.sleep(0.1)
                    continue

            CAMERA_RECONNECT_ATTEMPTS = 0  # Reset on successful frame
            current_time = time.time()

            
            # Process frame at intervals
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                # Get measurements from vision system
                annotated, measurements = measurement_app.process_frame(frame)
                
                # Get stitch count from serial
                current_stitch_count = serial_reader.get_stitch_count() if serial_reader else last_stitch_count


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

                    print(
                        ts() + 
                        f" 🔍 Raw measurements: "
                        f"seam={f'{raw_seam:.2f}' if raw_seam is not None else 'N/A'}mm, "
                        f"width={f'{raw_width:.2f}' if raw_width is not None else 'N/A'}mm"
                    )

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

                # For small/too-low measurements: ignore (do not confirm below lower bound)
                # If stitch width is above soft upper limit, check similarly
                if not valid_stitch and stitch_width_mm is not None and stitch_width_mm > stitch_upper_limit:
                    recent_w = [v for v in list(raw_width_history)[-CONFIRM_CONSECUTIVE:] if v is not None]
                    if len(recent_w) >= CONFIRM_CONSECUTIVE and all(v > stitch_upper_limit - CONFIRM_TOLERANCE_MM for v in recent_w):
                        valid_stitch = True
                        confirmed_override = True
                        print(ts() + f" 🚨Stitch width above {stitch_upper_limit}mm but sustained for {CONFIRM_CONSECUTIVE} samples - accepting as valid")

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
                            print(ts() + f" 📊 Using buffered average: seam={seam_length_mm:.2f}mm, "
                                f"width={stitch_width_mm:.2f}mm (from {len(valid_seam_buffer)} samples)")
                if stitch_delta > 0 and has_valid_measurement:
                    # Calculate moved distance
                    moved_distance_mm = stitch_delta * stitch_width_mm
                    total_distance_mm += moved_distance_mm
    

                    # Insert to database
                    if db:  
                        success = db.insert_measurement(
                            total_distance=round(total_distance_mm, 1),
                            stitch_length=round(stitch_width_mm, 1),
                            seam_allowance=round(seam_length_mm, 1)
                        )
                        if not success:
                            print(ts() + " ⚠️ Database insert failed - will retry on next valid measurement")
                    
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
            db.close()

        file_cleaner.stop() #stop file cleaner thread
        
        measurement_app.cap.release()

        if heartbeat:
            heartbeat.stop()
            
        cv2.destroyAllWindows()
        
        print(ts() + f" \n✅ Total frames processed: {frame_count}")
        print(ts() + f" 📁 Images saved to: {os.path.abspath(SAVE_DIR)}")
        print(ts() + " \n👋 System shutdown complete")


if __name__ == "__main__":
    main()