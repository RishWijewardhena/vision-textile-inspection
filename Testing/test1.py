
import cv2
import numpy as np

CAMERA_INDEX = '/dev/video0'  # Adjust if your camera is at a different index or path

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}")

window = "Exposure + Brightness + Contrast Editor"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)

# Trackbars:
# Exposure slider: 0..200  -> EV range approx -2.0 .. +2.0
# Brightness slider: 0..200 -> offset range -100 .. +100
# Contrast slider: 0..300   -> factor 0.0 .. 3.0 (1.0 is neutral)
cv2.createTrackbar("Exposure", window, 100, 200, lambda x: None)
cv2.createTrackbar("Brightness", window, 100, 200, lambda x: None)
cv2.createTrackbar("Contrast", window, 100, 300, lambda x: None)

def apply_adjustments(img01, exposure_slider, brightness_slider, contrast_slider):
    # Exposure in EV stops: factor = 2^EV
    ev = (exposure_slider - 100) / 50.0   # -2.0 .. +2.0
    exposure_factor = 2.0 ** ev

    # Brightness offset in normalized range
    brightness_offset = (brightness_slider - 100) / 255.0  # about -0.392 .. +0.392

    # Contrast factor around mid-gray point (0.5 in normalized image)
    contrast_factor = contrast_slider / 100.0  # 0.0 .. 3.0

    out = img01 * exposure_factor
    out = (out - 0.5) * contrast_factor + 0.5
    out = out + brightness_offset
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255).astype(np.uint8)
    return out

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: camera frame read failed")
        break

    frame01 = frame.astype(np.float32) / 255.0

    exposure = cv2.getTrackbarPos("Exposure", window)
    brightness = cv2.getTrackbarPos("Brightness", window)
    contrast = cv2.getTrackbarPos("Contrast", window)

    edited = apply_adjustments(frame01, exposure, brightness, contrast)
    cv2.imshow(window, edited)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("s"):
        cv2.imwrite("edited_output.jpg", edited)
        print("Saved: edited_output.jpg")
    elif key == 27 or key == ord("q"):  # ESC or q to quit
        break

cap.release()
cv2.destroyAllWindows()