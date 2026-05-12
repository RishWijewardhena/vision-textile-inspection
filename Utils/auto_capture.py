import cv2
import time
import os
from PIL import Image

def capture_camera(save_dir="Photos", interval=2):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    if not cap.isOpened():
        print("camera open failed")
        return

    print(f"Camera opened. Taking photo every {interval} seconds. Press Ctrl+C to stop.")

    try:
        count=1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filename = os.path.join(save_dir, f"photo_{count:03d}.jpg")
            
            # Save as RGB using PIL
            Image.fromarray(frame).save(filename)

            print(f"Saved {filename}")

            count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Capture stopped by user.")
    finally:
        cap.release()   

if __name__ == "__main__":
    capture_camera()