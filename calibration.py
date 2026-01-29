"""
Camera calibration module - intrinsics and extrinsics
"""

import cv2
import numpy as np
import json
import time
from config import (
    INTRINSICS_FILE, EXTRINSICS_FILE, DICT_TYPE,
    SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
    MIN_CHARUCO_CORNERS, CAMERA_INDEX,
    CALIB_W, CALIB_H, CAPTURE_DELAY, LOG_DEBUG, SHOW_WINDOWS
)

# --------------------------------------------------
# Load intrinsics
# --------------------------------------------------
def load_intrinsics(filepath=INTRINSICS_FILE):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64).ravel()

        if LOG_DEBUG:
            print(f"✅ Loaded intrinsics from {filepath}")

        return K, dist
    except Exception as e:
        raise FileNotFoundError(f"Failed to load intrinsics: {e}")

# --------------------------------------------------
# Save extrinsics
# --------------------------------------------------
def save_extrinsics(rvec, tvec, filepath=EXTRINSICS_FILE):
    data = {
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist()
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    if LOG_DEBUG:
        print(f"✅ Saved extrinsics to {filepath}")

# --------------------------------------------------
# Create ChArUco board
# --------------------------------------------------
def create_charuco_board():
    try:
        # OpenCV 4.7+
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        #(SQUARES_X, SQUARES_Y): Grid size (e.g., (5, 7) = 5 columns × 7 rows)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_Y, SQUARES_X),
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict
        )

        detector = cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters()
        )

        return board, detector

    except AttributeError:
        # OpenCV <= 4.6 fallback
        aruco_dict = cv2.aruco.Dictionary_get(DICT_TYPE)
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_Y, SQUARES_X,
            SQUARE_LENGTH, MARKER_LENGTH,
            aruco_dict
        )
        return board, None

# --------------------------------------------------
# Extrinsic calibration
# --------------------------------------------------
def run_extrinsic_calibration(board=None, charuco_detector=None, camera_index=CAMERA_INDEX):

    if board is None:
        board, charuco_detector = create_charuco_board()

    try:
        K, dist = load_intrinsics()
    except Exception as e:
        print(f"❌ {e}")
        return False

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_H)

    print("\n================ EXTRINSIC CALIBRATION ================")
    print(f"Need at least {MIN_CHARUCO_CORNERS} Charuco corners")
    print("Press SPACE to capture | ESC to cancel")
    print("======================================================\n")

    capture_start = None
    success_flag = False

    capture_start = time.time()
    print("⏳ Capturing in 5 seconds...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        #change the camera to  be able to detect the inverse charuco markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray=cv2.bitwise_not(gray)
        # cv2.imshow("Inverted Gray", inv_gray)

        charuco_corners = None
        charuco_ids = None

        # ---------- DETECTION ----------
        try:
            # OpenCV 4.7+
            charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(inv_gray)
            print(f"Detected {0 if charuco_ids is None else len(charuco_ids)} Charuco corners")

            if charuco_ids is None or len(charuco_ids) == 0:
                charuco_corners = None
                charuco_ids = None

        except AttributeError:
            print("❌ OpenCV version does not support CharucoDetector")
        # except Exception:
        #     # Older OpenCV fallback
        #     aruco_dict = cv2.aruco.Dictionary_get(DICT_TYPE)
        #     params = cv2.aruco.DetectorParameters_create()
        #     marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        #         inv_gray, aruco_dict, parameters=params
        #     )

        #     if marker_ids is not None:
        #         _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        #             marker_corners, marker_ids, gray, board
        #         )

        # ---------- DRAW ----------

        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                frame, charuco_corners, charuco_ids
            )

            count = len(charuco_ids)
            color = (0, 255, 0) if count >= MIN_CHARUCO_CORNERS else (0, 165, 255)
            cv2.putText(
                frame,
                f"Corners: {count}/{MIN_CHARUCO_CORNERS}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        else:
            cv2.putText(
                frame,
                "No ChArUco detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # cv2.imshow("Extrinsic Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if SHOW_WINDOWS:
            cv2.imshow("Inverted Gray", frame)

        # ---------- AUTO CAPTURE ----------
        if capture_start and (time.time() - capture_start) >= CAPTURE_DELAY:
            print("📸 Solving PnP...")

            # If we don't have enough corners by the time limit, fail fast
            if charuco_ids is None or len(charuco_ids) < MIN_CHARUCO_CORNERS:
                print("❌ Not enough ChArUco corners - calibration timed out")
                success_flag = False
                break

            obj_pts = board.getChessboardCorners()[charuco_ids.flatten()]
            img_pts = charuco_corners.reshape(-1, 2)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
                save_extrinsics(rvec, tvec)
                print("✅ Extrinsics saved")
                success_flag = True
                break
            else:
                print("❌ solvePnP failed - calibration unsuccessful")
                success_flag = False
                break

        # ---------- ESC ----------
        if key == 27:
            print("❌ Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensure window destruction is processed
    return success_flag

# --------------------------------------------------
# Test
# --------------------------------------------------
if __name__ == "__main__":
    board, detector = create_charuco_board()
    ok = run_extrinsic_calibration(board, detector)

    print("\nRESULT:", "SUCCESS ✅" if ok else "FAILED ❌")
