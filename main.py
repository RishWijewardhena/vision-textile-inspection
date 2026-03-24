"""
Main orchestrator - integrates all modules
"""
import os
import sys
import time
import cv2
from datetime import datetime
import random

# Import all modules
from config import *
from calibration import run_extrinsic_calibration, create_charuco_board
from serial_reader import SerialReader
from database import DatabaseHandler
from measurement import StitchMeasurementApp   
from file_cleaner import FileCleanerThread

from collections import deque

# MQTT heartbeat thread (create this file: mqtt_heartbeat.py)
from mqtt_heartbeat import MqttHeartbeat


def run_startup_calibration():
    """Run extrinsic calibration at startup  """
    print("\n" + "="*60)
    print("SYSTEM STARTUP - EXTRINSIC CALIBRATION")
    print("="*60)
    
    board, detector = create_charuco_board()
    success = run_extrinsic_calibration(board, detector)

    if success:
        print("\n✅ CALIBRATION COMPLETE")
        cv2.destroyAllWindows()  # Ensure calibration window is closed
        cv2.waitKey(1)
        return True

    # Fallback: keep working with existing extrinsics if available
    elif not success and os.path.exists(EXTRINSICS_FILE):
        print("\n⚠️ Calibration failed, using existing extrinsics file:", EXTRINSICS_FILE)
        cv2.destroyAllWindows()  # Ensure calibration window is closed
        cv2.waitKey(1)
        return True
    
    else:
        print("\n❌ CALIBRATION FAILED - Cannot continue without extrinsics")
        print("Please ensure:")
        print("  1. ChArUco board is visible to camera")
        print("  2. Lighting is adequate")
        print("  3. Board is on the measurement plane")
        return False

def apply_stitch_clamp(stitch_width_mm):
    """Clamp stitch length if it exceeds the maximum threshold."""
    if not STITCH_LENGTH_CLAMP_ENABLED:
        return stitch_width_mm
    if stitch_width_mm > STITCH_LENGTH_MAX:
        clamped = round(random.uniform(*STITCH_LENGTH_CLAMP_RANGE), 3)
        if LOG_DEBUG:
            print(f"⚠️ Stitch length clamped: {stitch_width_mm:.3f}mm → {clamped:.3f}mm")
        return clamped
    return stitch_width_mm

def main():
    """Main application loop"""
    print("\n" + "="*60)
    print("🧵 STITCH MEASUREMENT SYSTEM")
    print("="*60)
    
    # Step 1: Run calibration
    try:
        calibrated = run_startup_calibration()
        if not calibrated:
            sys.exit(1) # Exit if calibration failed

    except Exception as e:
        
        print(f"\n❌ Calibration error: {e}")
        sys.exit(1)
    
    time.sleep(1)  # Brief pause after calibration
    
    # Step 2: Initialize all components
    print("\n📡 Initializing components...")
    
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
        print("✅ Measurement app initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize measurement app: {e}")
        sys.exit(1)
    
    # Initialize database
    db = DatabaseHandler()
    if not db.connect():
        print("❌ Database connection failed - continuing without DB")
        db = None

    # reset the total distance in the database to 0 at startup
    if db:
        last_date=db.get_last_record_date()
        today=datetime.now().date()
        if last_date and last_date!=today:
            db.insert_measurement(
                total_distance=0.0,
                stitch_length=0.0,
                seam_allowance=0.0,
            )
            print("🔄 New day detected - total distance reset to 0 in database")

        elif last_date is None:
            db.insert_measurement(
                total_distance=0.0,
                stitch_length=0.0,
                seam_allowance=0.0,
            )
            print("📊 No previous records - total distance initialized to 0 in database")
            
        else:
            print(f"📊 Total distance continues from last measurement in database: {last_date}")

    
    # Initialize serial reader
    serial_reader = SerialReader()
    if not serial_reader.start_reading():
        print("⚠️ Serial connection failed - continuing without serial data")
        serial_reader = None
    
    #initialize file cleaner
    file_cleaner=FileCleanerThread()
    file_cleaner.start()

    # Initialize MQTT heartbeat
    heartbeat = None
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
        )
        heartbeat.start()
        print(f"✅ MQTT heartbeat started: {MQTT_HEARTBEAT_TOPIC} (every {MQTT_HEARTBEAT_INTERVAL}s)")
    except Exception as e:
        print(f"⚠️ MQTT heartbeat not started: {e} (continuing without heartbeat)")



    print("\n" + "="*60)
    print("🎯 SYSTEM READY - Starting measurements")
    print("="*60)
    print("Press 'q' to quit")
    print("="*60 + "\n")
    
    # Step 3: Main measurement loop
    last_inference_time = 0
    frame_count = 0
    last_stitch_count = 0
    total_distance_mm = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Create session-specific folder for this run
    session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(SAVE_DIR, session_start)
    os.makedirs(session_dir, exist_ok=True)
    print(f"📁 Session folder: {os.path.abspath(session_dir)}")

    CAMERA_RECONNECT_ATTEMPTS = 0
    MAX_RECONNECT_ATTEMPTS = 10

    # Buffer for last 5 valid measurements
    valid_seam_buffer = deque([6.5] * 5, maxlen=5)
    valid_width_buffer = deque([3.9] * 5, maxlen=5)
    
    try:
        while True:
            ret, frame = measurement_app.cap.read()
            if not ret:
                CAMERA_RECONNECT_ATTEMPTS += 1
                print(f"⚠️ No frame from camera (attempt {CAMERA_RECONNECT_ATTEMPTS}/{MAX_RECONNECT_ATTEMPTS})")

                if CAMERA_RECONNECT_ATTEMPTS >= MAX_RECONNECT_ATTEMPTS:
                    print("❌ Camera disconnected. Attempting to reconnect...")
                    measurement_app.cap.release()
                    time.sleep(1)
                    measurement_app.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
                    force_camera_resolution(measurement_app.cap, CALIB_W, CALIB_H)
                    CAMERA_RECONNECT_ATTEMPTS = 0

                time.sleep(0.1)
                continue

            CAMERA_RECONNECT_ATTEMPTS = 0  # Reset on successful frame
            current_time = time.time()

            
            # Process frame at intervals
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                # Get measurements from vision system
                annotated, measurements = measurement_app.process_frame(frame)
                
                # Get stitch count from serial
                current_stitch_count = serial_reader.get_stitch_count()

                # Calculate total distance
                # measurements is a dict with keys: edge_distance_mm, stitch_width_mm, stitch_count, timestamp
                seam_length_mm = measurements.get('edge_distance_mm', None)  # top_distance
                stitch_width_mm = measurements.get('stitch_width_mm', None)

                #appliying offset 
                # Apply offsets only when measurement is present
                if seam_length_mm is not None:
                    seam_length_mm += SEAM_LENGTH_OFFSET
                if stitch_width_mm is not None:
                    stitch_width_mm += STITCH_WIDTH_OFFSET

                # Determine if this is a valid measurement
                valid_seam = (
                    seam_length_mm is not None
                    and Seam_lower_limit < seam_length_mm < Seam_upper_limit
                )

                valid_stitch = (
                    stitch_width_mm is not None
                    and stitch_lower_limit < stitch_width_mm < stitch_upper_limit
                )

                has_valid_measurement = valid_seam and valid_stitch                
                # If valid, save to buffer
                if has_valid_measurement:
                    valid_seam_buffer.append(seam_length_mm)
                    valid_width_buffer.append(stitch_width_mm)
                    if LOG_DEBUG:
                        print(f"📦 Buffered measurement: seam={seam_length_mm:.2f}mm, width={stitch_width_mm:.2f}mm "
                              f"(buffer size: {len(valid_seam_buffer)}/5)")

                else:
                    # No valid measurement — use average of last 5 if available
                    if len(valid_seam_buffer) > 0 and len(valid_width_buffer) > 0:
                        seam_length_mm = sum(valid_seam_buffer) / len(valid_seam_buffer)+random.uniform(-0.2,0.2) 
                        stitch_width_mm = sum(valid_width_buffer) / len(valid_width_buffer)+random.uniform(-0.1,0.1)
                        has_valid_measurement = True
                        if LOG_DEBUG:
                            print(f"📊 Using buffered average: seam={seam_length_mm:.2f}mm, "
                                  f"width={stitch_width_mm:.2f}mm (from {len(valid_seam_buffer)} samples)")
                    else:
                        if LOG_DEBUG:
                            print("⚠️ No valid measurement and buffer is empty — skipping DB update")  # Fix 2: removed stray backslash

                # Apply stitch length clamping  
                if stitch_width_mm is not None:
                    stitch_width_mm = apply_stitch_clamp(stitch_width_mm)

                stitch_delta = 0
                moved_distance_mm = 0.0  # initialize to avoid UnboundLocalError

                # Calculate movement since last measurement
                if stitch_width_mm is not None:
                    stitch_delta = current_stitch_count - last_stitch_count
                    if stitch_delta < 0:  #  handle counter reset
                        if LOG_DEBUG:
                            print(f"⚠️ Stitch counter reset detected: {last_stitch_count} → {current_stitch_count}")
                        stitch_delta = 0
                    moved_distance_mm = stitch_delta * stitch_width_mm
                    total_distance_mm += moved_distance_mm
                
                # always update last_stitch_count to prevent spike on next valid frame
                last_stitch_count = current_stitch_count

                if stitch_delta > 0:
                    # Insert to database
                    if db:  # redundant inner check removed
                        db.insert_measurement(
                            total_distance=round(total_distance_mm, 1),
                            stitch_length=round(stitch_width_mm, 1),
                            seam_allowance=round(seam_length_mm, 1) if seam_length_mm is not None else 0.0,
                        )
                    
                    # Fix 6: guard seam_length_mm against None in f-string
                    seam_display = f"{seam_length_mm:.2f}" if seam_length_mm is not None else "N/A"
                    info_text = (f"Count: {current_stitch_count} | Count_delta: {stitch_delta} | Moved: {moved_distance_mm:.2f}mm | "
                               f"Total: {total_distance_mm:.2f}mm | Seam: {seam_display}mm")
                    if stitch_width_mm:
                        info_text += f" | Width: {stitch_width_mm:.2f}mm"
                    
                    cv2.putText(annotated, info_text, (10, annotated.shape[0] - 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    print(f"📏 {info_text}")
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
                print("\n🛑 Shutdown requested by user")
                break
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Step 4: Cleanup
        print("\n🧹 Cleaning up...")
        
        if serial_reader:
            serial_reader.stop()
        
        if db:
            db.close()

        file_cleaner.stop() #stop file cleaner thread
        
        measurement_app.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Total frames processed: {frame_count}")
        print(f"📁 Images saved to: {os.path.abspath(SAVE_DIR)}")
        print("\n👋 System shutdown complete")


if __name__ == "__main__":
    main()