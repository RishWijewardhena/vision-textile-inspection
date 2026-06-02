import os
import math
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from config import (
    NEEDLE_ANGLE_MODEL_PATH,
    NEEDLE_ANGLE_CONF_THRESH,
    NEEDLE_ANGLE_IOU_THRESH,
    NEEDLE_NOT_ROTATED_ANGLE_MIN,
    NEEDLE_NOT_ROTATED_ANGLE_MAX,
)

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH  = NEEDLE_ANGLE_MODEL_PATH
IMAGES_DIR  = "test_images"
OUTPUT_DIR  = "results_angles"
CONF_THRESH = NEEDLE_ANGLE_CONF_THRESH
IOU_THRESH  = NEEDLE_ANGLE_IOU_THRESH
IMGSZ       = 640
DEFAULT_NOT_ROTATED_ANGLE_MIN = NEEDLE_NOT_ROTATED_ANGLE_MIN
DEFAULT_NOT_ROTATED_ANGLE_MAX = NEEDLE_NOT_ROTATED_ANGLE_MAX

# Colors (BGR)
COLOR_BOX       = (0, 255, 0)     # Green  — bounding box
COLOR_HORIZ_REF = (255, 100, 0)   # Blue   — horizontal reference line
COLOR_VERT_REF  = (0, 100, 255)   # Orange — vertical reference line
COLOR_AXIS      = (0, 220, 255)   # Yellow — box major axis
COLOR_ARC_H     = (255, 200, 0)   # Cyan arc for horizontal angle
COLOR_ARC_V     = (0, 200, 255)   # Orange arc for vertical angle
COLOR_LABEL     = (255, 255, 255) # White text
# ──────────────────────────────────────────────────────────


def calculate_needle_angles(angle_rad):
    angle_deg = math.degrees(angle_rad)
    angle_deg = ((angle_deg + 90) % 180) - 90
    horiz_angle = abs(angle_deg)
    vert_angle = 90.0 - horiz_angle
    orientation_angle = math.degrees(angle_rad) % 180.0
    return angle_deg, horiz_angle, vert_angle, orientation_angle


def draw_angle_annotation(img, center, angle_rad, ref_len):
    """
    Draw:
      - Horizontal reference ray (→) in blue
      - Vertical reference ray (↓) in orange
      - Box major-axis ray in yellow
      - Arc + label for horizontal angle (axis vs horizontal)
      - Arc + label for vertical angle (axis vs vertical)
    """
    cx, cy = center

    angle_deg, horiz_angle, vert_angle, _ = calculate_needle_angles(angle_rad)

    # ── Reference rays ────────────────────────────────────
    ray = int(ref_len * 0.7)
    # Horizontal ray (rightward)
    cv2.arrowedLine(img, (cx, cy), (cx + ray, cy),
                    COLOR_HORIZ_REF, 1, cv2.LINE_AA, tipLength=0.12)
    # Vertical ray (downward, image coords)
    cv2.arrowedLine(img, (cx, cy), (cx, cy + ray),
                    COLOR_VERT_REF, 1, cv2.LINE_AA, tipLength=0.12)

    # ── Box major-axis ray ────────────────────────────────
    rad = math.radians(angle_deg)          # use normalised angle
    axis_ex = int(cx + ray * math.cos(rad))
    axis_ey = int(cy + ray * math.sin(rad))
    cv2.arrowedLine(img, (cx, cy), (axis_ex, axis_ey),
                    COLOR_AXIS, 2, cv2.LINE_AA, tipLength=0.12)

    # ── Arcs ──────────────────────────────────────────────
    arc_r = int(ray * 0.45)

    # Horizontal arc: from 0° to angle_deg (image-coordinate convention)
    a_start = min(0.0, angle_deg)
    a_end   = max(0.0, angle_deg)
    cv2.ellipse(img, (cx, cy), (arc_r, arc_r), 0,
                a_start, a_end, COLOR_ARC_H, 1, cv2.LINE_AA)

    # Vertical arc: from 90° toward angle_deg
    v_start = min(90.0, 90.0 + angle_deg if angle_deg < 0 else angle_deg)
    v_end   = max(angle_deg, 90.0)
    cv2.ellipse(img, (cx, cy), (arc_r, arc_r), 0,
                v_start, v_end, COLOR_ARC_V, 1, cv2.LINE_AA)

    # ── Angle labels ──────────────────────────────────────
    # Place horizontal label along the arc bisector (~half of horiz_angle)
    mid_h_rad = math.radians(angle_deg / 2.0)
    lx_h = cx + int((arc_r + 10) * math.cos(mid_h_rad))
    ly_h = cy + int((arc_r + 10) * math.sin(mid_h_rad))
    cv2.putText(img, f"H:{horiz_angle:.1f}",
                (lx_h - 5, ly_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ARC_H, 1, cv2.LINE_AA)

    # Place vertical label along the arc bisector (~midway between axis and vertical)
    mid_v_deg = (angle_deg + 90.0) / 2.0
    mid_v_rad = math.radians(mid_v_deg)
    lx_v = cx + int((arc_r + 10) * math.cos(mid_v_rad))
    ly_v = cy + int((arc_r + 10) * math.sin(mid_v_rad))
    cv2.putText(img, f"V:{vert_angle:.1f}",
                (lx_v - 5, ly_v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ARC_V, 1, cv2.LINE_AA)

    return horiz_angle, vert_angle


class NeedleAngleDetector:
    def __init__(
        self,
        model_path=MODEL_PATH,
        not_rotated_angle_min=DEFAULT_NOT_ROTATED_ANGLE_MIN,
        not_rotated_angle_max=DEFAULT_NOT_ROTATED_ANGLE_MAX,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH,
        imgsz=IMGSZ,
    ):
        self.model = YOLO(str(model_path))
        self.not_rotated_angle_min = float(not_rotated_angle_min)
        self.not_rotated_angle_max = float(not_rotated_angle_max)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

    def measure_frame(self, frame, annotate=False):
        if frame is None:
            return {
                "rotated": False,
                "detections": [],
                "h_angle": None,
                "v_angle": None,
                "orientation_angle": None,
                "error": "No frame provided",
            }

        draw_img = frame.copy() if annotate else None
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )

        result = results[0]
        boxes = result.obb
        if boxes is None or len(boxes) == 0:
            response = {
                "rotated": False,
                "detections": [],
                "h_angle": None,
                "v_angle": None,
                "orientation_angle": None,
            }
            if annotate:
                response["annotated"] = draw_img
            return response

        detections = []
        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = self.model.names[cls_id]
            conf = float(box.conf.item())
            xywhr = box.xywhr[0].cpu().numpy()
            cx, cy, w, h, angle_rad = (
                float(xywhr[0]),
                float(xywhr[1]),
                float(xywhr[2]),
                float(xywhr[3]),
                float(xywhr[4]),
            )
            angle_deg, h_ang, v_ang, orientation_ang = calculate_needle_angles(angle_rad)
            rotated = not (self.not_rotated_angle_min <= orientation_ang <= self.not_rotated_angle_max)
            detection = {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "angle_deg": angle_deg,
                "h_angle": h_ang,
                "v_angle": v_ang,
                "orientation_angle": orientation_ang,
                "rotated": rotated,
            }
            detections.append(detection)

            if annotate:
                center = (int(cx), int(cy))
                ref_len = int(max(w, h) * 0.65)
                corners = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.int32)
                cv2.polylines(draw_img, [corners], isClosed=True, color=COLOR_BOX, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(draw_img, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
                draw_angle_annotation(draw_img, center, angle_rad, ref_len)

        primary = max(detections, key=lambda item: item["confidence"])
        response = {
            "rotated": any(item["rotated"] for item in detections),
            "detections": detections,
            "h_angle": primary["h_angle"],
            "v_angle": primary["v_angle"],
            "confidence": primary["confidence"],
            "orientation_angle": primary["orientation_angle"],
        }
        if annotate:
            response["annotated"] = draw_img
        return response


def _ts():
    return datetime.now().strftime("[%H:%M:%S]")


class NeedleAngleWorker(threading.Thread):
    def __init__(self, model_path, interval_sec, not_rotated_angle_min, not_rotated_angle_max, annotation_dir=None):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.interval_sec = interval_sec
        self.not_rotated_angle_min = not_rotated_angle_min
        self.not_rotated_angle_max = not_rotated_angle_max
        self.annotation_dir = annotation_dir
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

    def set_annotation_dir(self, annotation_dir):
        os.makedirs(annotation_dir, exist_ok=True)
        with self._lock:
            self.annotation_dir = annotation_dir

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
                        not_rotated_angle_min=self.not_rotated_angle_min,
                        not_rotated_angle_max=self.not_rotated_angle_max,
                    )

                with self._lock:
                    annotation_dir = self.annotation_dir

                result = detector.measure_frame(frame, annotate=bool(annotation_dir))
                annotated = result.pop("annotated", None)
                result["checked_at"] = time.time()

                if annotation_dir and annotated is not None:
                    filename = "needle_angle_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    save_path = os.path.join(annotation_dir, filename)
                    if cv2.imwrite(save_path, annotated):
                        result["annotation_path"] = save_path
                        print(_ts() + f" 🖼️ Needle angle annotation saved: {save_path}")
                    else:
                        print(_ts() + f" ⚠️ Needle angle annotation save failed: {save_path}")

                orientation_angle = result.get("orientation_angle")
                angle_text = f"{orientation_angle:.1f}" if orientation_angle is not None else "N/A"
                rotated = result.get("rotated", False)
                print(_ts() + f" 🧭 Needle angle checked: angle={angle_text}°, rotated={rotated}")

            except Exception as exc:
                result = {
                    "rotated": False,
                    "detections": [],
                    "h_angle": None,
                    "v_angle": None,
                    "orientation_angle": None,
                    "error": str(exc),
                    "checked_at": time.time(),
                }
                print(_ts() + f" ⚠️ Needle angle inference failed: {exc}")

            with self._lock:
                self._latest_result = result
                self._busy = False

    def stop(self):
        self._stop_event.set()
        self._frame_ready.set()


def process_image(img_path: Path, model: YOLO, output_dir: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  Could not read {img_path.name}, skipping.")
        return

    results = model.predict(
        source=str(img_path),
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMGSZ,
        verbose=False,
    )

    result = results[0]
    boxes  = result.obb

    if boxes is None or len(boxes) == 0:
        print(f"  [{img_path.name}] — no detections")
        cv2.imwrite(str(output_dir / img_path.name), img)
        return

    print(f"  [{img_path.name}] — {len(boxes)} detection(s)")

    for i, box in enumerate(boxes):
        cls_id   = int(box.cls.item())
        cls_name = model.names[cls_id]
        conf     = float(box.conf.item())

        # xywhr: [cx, cy, w, h, angle_radians]
        xywhr   = box.xywhr[0].cpu().numpy()
        cx, cy, w, h, angle_rad = float(xywhr[0]), float(xywhr[1]), \
                                   float(xywhr[2]), float(xywhr[3]), float(xywhr[4])
        center  = (int(cx), int(cy))
        ref_len = int(max(w, h) * 0.65)

        # 4 corners of the OBB
        corners = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(np.int32)

        # Draw oriented bounding box
        cv2.polylines(img, [corners], isClosed=True,
                      color=COLOR_BOX, thickness=2, lineType=cv2.LINE_AA)

        # Draw center dot
        cv2.circle(img, center, 4, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw angle annotation
        h_ang, v_ang = draw_angle_annotation(img, center, angle_rad, ref_len)

        # Class + confidence label above the box
        tx = int(corners[:, 0].min())
        ty = int(corners[:, 1].min()) - 8
        if ty < 15:
            ty = int(corners[:, 1].max()) + 18

        label = f"{cls_name} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (tx - 2, ty - lh - 4), (tx + lw + 2, ty + 2),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 2, cv2.LINE_AA)

        print(f"    [{i+1}] {cls_name:<15} conf:{conf:.2f}  "
              f"H-angle:{h_ang:6.1f}°  V-angle:{v_ang:6.1f}°")

    cv2.imwrite(str(output_dir / img_path.name), img)


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = [p for ext in exts for p in Path(IMAGES_DIR).glob(ext)]

    if not image_files:
        print(f"No images found in '{IMAGES_DIR}/'")
        return

    print(f"\nProcessing {len(image_files)} image(s)...\n")
    for img_path in sorted(image_files):
        process_image(img_path, model, output_dir)

    print(f"\nDone! Annotated images saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
