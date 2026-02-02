"""
Main orchestrator - integrates all modules
"""
import os
import sys
import time
import cv2
from datetime import datetime

# Import all modules
from config import *
from calibration import run_extrinsic_calibration, create_charuco_board
from serial_reader import SerialReader
from database import DatabaseHandler
from measurement import StitchMeasurementApp   
from file_cleaner import FileCleanerThread


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
    else:
        db.ensure_table_exists()
    
    # Initialize serial reader
    serial_reader = SerialReader()
    if not serial_reader.start_reading():
        print("⚠️ Serial connection failed - continuing without serial data")
        serial_reader = None
    
    #initialize file cleaner
    file_cleaner=FileCleanerThread()
    file_cleaner.start()


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
    
    try:
        while True:
            ret, frame = measurement_app.cap.read()
            if not ret:
                print("⚠️ No frame from camera, retrying...")
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # Process frame at intervals
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                # Get measurements from vision system
                annotated, measurements = measurement_app.process_frame(frame)
                
                # Get stitch count from serial
                current_stitch_count = serial_reader.get_stitch_count() if serial_reader else 0

              
                
                # Calculate total distance
                # measurements is a dict with keys: edge_distance_mm, stitch_width_mm, stitch_count, timestamp
                seam_length_mm = measurements.get('edge_distance_mm', None)  # top_distance
                stitch_width_mm = measurements.get('stitch_width_mm', None)

                # Calculate movement since last measurement
                if stitch_width_mm is not None:
                    stitch_delta = current_stitch_count - last_stitch_count
                    moved_distance_mm = stitch_delta * stitch_width_mm
                    total_distance_mm += moved_distance_mm
                    last_stitch_count = current_stitch_count
                
                if seam_length_mm is not None and current_stitch_count > 0 and stitch_width_mm is not None:
                    
                    
                    # Insert to database
                    if db and stitch_delta > 0: #only log if there's a new rotation
                        db.insert_measurement(
                            total_distance=total_distance_mm,
                            stitch_length=stitch_width_mm,
                            seam_allowance=seam_length_mm,
                        )
                    
                    # Display info on frame
                    info_text = (f"Count: {current_stitch_count} | Count_delta: {stitch_delta} | Moved: {moved_distance_mm:.2f}mm | "
                               f"Total: {total_distance_mm:.2f}mm | Seam: {seam_length_mm:.2f}mm")
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
                save_path = os.path.join(SAVE_DIR, f"frame_{frame_count:05d}_{timestamp}.jpg")
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