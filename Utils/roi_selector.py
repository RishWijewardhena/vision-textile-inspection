import argparse
import ast
from pathlib import Path

import cv2


WINDOW_NAME = "ROI Selector"
DEFAULT_CALIB_W = 1280
DEFAULT_CALIB_H = 960
DEFAULT_ROI_X_MIN = 100
DEFAULT_ROI_Y_MIN = 300


def load_calibration_size():
    config_path = Path(__file__).resolve().parent.parent / "config.py"
    try:
        source = config_path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(config_path))
    except OSError:
        return DEFAULT_CALIB_W, DEFAULT_CALIB_H

    values = {}
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"CALIB_W", "CALIB_H"}:
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                values[target.id] = node.value.value

    return (
        values.get("CALIB_W", DEFAULT_CALIB_W),
        values.get("CALIB_H", DEFAULT_CALIB_H),
    )


CALIB_W, CALIB_H = load_calibration_size()


class ROISelector:
    def __init__(self, image):
        self.base_image = image
        self.display_image = image.copy()
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.selection = None

    def _normalize_rect(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        return left, top, right, bottom

    def _redraw(self, preview_point=None):
        self.display_image = self.base_image.copy()
        if self.selection is not None:
            x1, y1, x2, y2 = self.selection
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (144, 238, 144), 2)
            cv2.putText(
                self.display_image,
                f"ROI: x=({x1},{x2}) y=({y1},{y2})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if self.drawing and self.start_point is not None and preview_point is not None:
            x1, y1, x2, y2 = self._normalize_rect(self.start_point, preview_point)
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            self.display_image,
            "Drag to select ROI | s: save to .env | r: reset | q: quit",
            (10, self.display_image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.drawing = True
            self._redraw(preview_point=self.end_point)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self._redraw(preview_point=self.end_point)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.end_point = (x, y)
            self.drawing = False
            rect = self._normalize_rect(self.start_point, self.end_point)
            if rect[0] != rect[2] and rect[1] != rect[3]:
                self.selection = rect
            self._redraw()


def _upsert_env_values(env_path, updates, section_header="# ROI Settings"):
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

    seen = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if "=" in line and not stripped.startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        new_lines.append(line)

    if new_lines and new_lines[-1] != "":
        new_lines.append("")

    if not any(line.strip() == section_header for line in new_lines):
        new_lines.append(section_header)

    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def ensure_roi_env_defaults(env_path, image_shape):
    height, width = image_shape[:2]
    updates = {
        "ROI_ENABLED": "true",
        "ROI_X_MIN": str(DEFAULT_ROI_X_MIN),
        "ROI_X_MAX": str(max(DEFAULT_ROI_X_MIN + 1, width - 10)),
        "ROI_Y_MIN": str(DEFAULT_ROI_Y_MIN),
        "ROI_Y_MAX": str(max(DEFAULT_ROI_Y_MIN + 1, height - 200)),
    }
    _upsert_env_values(env_path, updates)


def update_env_file(env_path, roi):
    x1, y1, x2, y2 = roi
    updates = {
        "ROI_ENABLED": "true",
        "ROI_X_MIN": str(x1),
        "ROI_X_MAX": str(x2),
        "ROI_Y_MIN": str(y1),
        "ROI_Y_MAX": str(y2),
    }
    _upsert_env_values(env_path, updates)


def capture_frame(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_H)

    frozen_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read a frame from the camera")

            if frame.shape[1] != CALIB_W or frame.shape[0] != CALIB_H:
                raise RuntimeError(
                    f"Camera returned {frame.shape[1]}x{frame.shape[0]}, expected "
                    f"{CALIB_W}x{CALIB_H}. Fix the camera resolution before picking ROI."
                )

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Resolution {CALIB_W}x{CALIB_H} | space: capture | q: quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(30) & 0xFF
            if key == ord(" "):
                frozen_frame = frame.copy()
                break
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()

    return frozen_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture or load an image, draw an ROI, and save it to .env."
    )
    parser.add_argument(
        "--image",
        help="Optional path to an image file. If omitted, the camera preview is used.",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index or device path to use when --image is not provided.",
    )
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parent.parent / ".env"),
        help="Path to the .env file to update.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_path = Path(args.env_file).resolve()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise RuntimeError(f"Could not read image: {args.image}")
    else:
        camera_value = int(args.camera) if str(args.camera).isdigit() else args.camera
        image = capture_frame(camera_value)
        if image is None:
            print("No frame captured. Exiting without changes.")
            return

    ensure_roi_env_defaults(env_path, image.shape)

    selector = ROISelector(image)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, selector.mouse_callback)
    selector._redraw()

    while True:
        cv2.imshow(WINDOW_NAME, selector.display_image)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            selector.selection = None
            selector.start_point = None
            selector.end_point = None
            selector.drawing = False
            selector._redraw()
        elif key == ord("s"):
            if selector.selection is None:
                print("Select an ROI before saving.")
                continue
            update_env_file(env_path, selector.selection)
            x1, y1, x2, y2 = selector.selection
            print(f"Saved ROI to {env_path}")
            print("Use these values in config:")
            print(f"ROI_ENABLED=true")
            print(f"ROI_X_MIN={x1}")
            print(f"ROI_X_MAX={x2}")
            print(f"ROI_Y_MIN={y1}")
            print(f"ROI_Y_MAX={y2}")
            break
        elif key == ord("q") or key == 27:
            print("Exited without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
