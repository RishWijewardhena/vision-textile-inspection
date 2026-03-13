"""Live YOLO segmentation viewer.

This utility loads the YOLO model path and camera resolution from `config.py`
and shows a live annotated feed with segmentation masks, contours, boxes, and
confidence labels.

Inference runs every 0.5 seconds. The first two annotated inference frames are
saved automatically into a `photos_checked` folder in the current directory.

Controls:
	- q / ESC: quit
	- s: save the current annotated frame
"""

from __future__ import annotations

import ast
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = PROJECT_ROOT / "config.py"
SAVE_DIR = CURRENT_DIR / "photos_checked"
INFERENCE_INTERVAL_SECONDS = 0.5
AUTO_SAVE_COUNT = 500

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ViewerSettings:
	"""Runtime settings used by the live viewer."""

	model_path: Path
	camera_index: Any
	calib_w: int
	calib_h: int
	conf_thresh: float = 0.20
	iou_thresh: float = 0.45
	max_detections: int = 200


def _literal_from_ast(node: ast.AST) -> Any:
	"""Safely evaluate simple literals from an AST node."""
	try:
		return ast.literal_eval(node)
	except Exception:
		return None


def _parse_config_defaults(config_path: Path) -> dict[str, Any]:
	"""Parse simple values from `config.py` without importing the module."""
	defaults: dict[str, Any] = {
		"MODEL_PATH": "best_Model.pt",
		"CALIB_W": 640,
		"CALIB_H": 640,
		"CONF_THRESH": 0.20,
		"IOU_THRESH": 0.45,
		"MAX_DETECTIONS": 200,
		"CAMERA_INDEX": 0,
	}

	if not config_path.exists():
		return defaults

	tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
	for node in tree.body:
		if not isinstance(node, ast.Assign):
			continue
		for target in node.targets:
			if not isinstance(target, ast.Name):
				continue
			key = target.id
			if key not in defaults:
				continue

			value = _literal_from_ast(node.value)
			if value is not None:
				defaults[key] = value

	return defaults


def _best_effort_camera_index(parsed_defaults: dict[str, Any]) -> Any:
	"""Resolve the camera source when full `config.py` import is unavailable."""
	camera_index = parsed_defaults.get("CAMERA_INDEX", 0)
	if camera_index != 0:
		return camera_index

	try:
		hardware_utils = importlib.import_module("hardware_utils")
		finder = getattr(hardware_utils, "find_camera", None)
		if callable(finder):
			detected = finder()
			if detected is not None:
				return detected
	except Exception:
		pass

	return camera_index


def load_settings() -> ViewerSettings:
	"""Load model path and camera resolution from `config.py`.

	The project `config.py` validates DB environment variables during import.
	If that import fails, this function falls back to parsing only the values
	needed for the viewer.
	"""
	parsed_defaults = _parse_config_defaults(CONFIG_FILE)

	model_path = PROJECT_ROOT / str(parsed_defaults["MODEL_PATH"])
	camera_index = _best_effort_camera_index(parsed_defaults)
	calib_w = int(parsed_defaults["CALIB_W"])
	calib_h = int(parsed_defaults["CALIB_H"])
	conf_thresh = float(parsed_defaults["CONF_THRESH"])
	iou_thresh = float(parsed_defaults["IOU_THRESH"])
	max_detections = int(parsed_defaults["MAX_DETECTIONS"])

	try:
		config = importlib.import_module("config")
		model_path = PROJECT_ROOT / str(getattr(config, "MODEL_PATH", model_path.name))
		camera_index = getattr(config, "CAMERA_INDEX", camera_index)
		calib_w = int(getattr(config, "CALIB_W", calib_w))
		calib_h = int(getattr(config, "CALIB_H", calib_h))
		conf_thresh = float(getattr(config, "CONF_THRESH", conf_thresh))
		iou_thresh = float(getattr(config, "IOU_THRESH", iou_thresh))
		max_detections = int(getattr(config, "MAX_DETECTIONS", max_detections))
	except Exception as exc:
		print(f"[WARN] Could not fully import config.py: {exc}")
		print("[WARN] Falling back to parsed values for model path and resolution.")

	return ViewerSettings(
		model_path=model_path,
		camera_index=camera_index,
		calib_w=calib_w,
		calib_h=calib_h,
		conf_thresh=conf_thresh,
		iou_thresh=iou_thresh,
		max_detections=max_detections,
	)


def build_color(class_id: int) -> tuple[int, int, int]:
	"""Create a stable BGR color for each class id."""
	palette = [
		(0, 255, 0),
		(0, 165, 255),
		(255, 0, 255),
		(255, 255, 0),
		(0, 255, 255),
		(255, 128, 0),
	]
	return palette[class_id % len(palette)]


def get_instance_mask(result: Any, idx: int, height: int, width: int) -> np.ndarray | None:
	"""Extract one instance segmentation mask as a full-size binary bitmap."""
	if getattr(result, "masks", None) is None:
		return None

	try:
		data = result.masks.data
		mask = data[idx].cpu().numpy() if hasattr(data[idx], "cpu") else np.array(data[idx])
		if mask.shape != (height, width):
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
		mask = (mask > 0.5).astype(np.uint8)
		if np.count_nonzero(mask) > 0:
			return mask
	except Exception:
		pass

	try:
		polygon = np.array(result.masks.xy[idx], dtype=np.int32)
		mask = np.zeros((height, width), dtype=np.uint8)
		if polygon.ndim == 2 and polygon.shape[0] >= 3:
			cv2.fillPoly(mask, [polygon], 1)
			if np.count_nonzero(mask) > 0:
				return mask
	except Exception:
		pass

	return None


def annotate_result(frame: np.ndarray, result: Any, names: dict[int, str] | list[str] | None) -> np.ndarray:
	"""Draw masks, contours, boxes, and labels on the frame."""
	annotated = frame.copy()
	overlay = frame.copy()
	height, width = frame.shape[:2]

	if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
		return annotated

	try:
		boxes = result.boxes.xyxy.cpu().numpy()
		classes = result.boxes.cls.cpu().numpy().astype(int)
		scores = result.boxes.conf.cpu().numpy()
	except Exception:
		boxes = np.array(result.boxes.xyxy)
		classes = np.array(result.boxes.cls).astype(int)
		scores = np.array(result.boxes.conf)

	for idx, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
		x1, y1, x2, y2 = map(int, box)
		color = build_color(int(class_id))
		mask = get_instance_mask(result, idx, height, width)

		if mask is not None:
			overlay[mask > 0] = color
			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(annotated, contours, -1, color, 2)

		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

		if isinstance(names, dict):
			label_name = names.get(int(class_id), str(class_id))
		elif isinstance(names, list) and 0 <= int(class_id) < len(names):
			label_name = names[int(class_id)]
		else:
			label_name = str(class_id)

		label = f"{label_name} {float(score):.2f}"
		(text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
		text_y = max(20, y1 - 8)
		cv2.rectangle(
			annotated,
			(x1, text_y - text_h - baseline - 4),
			(x1 + text_w + 8, text_y + 4),
			color,
			-1,
		)
		cv2.putText(
			annotated,
			label,
			(x1 + 4, text_y - 2),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 0, 0),
			2,
			cv2.LINE_AA,
		)

	annotated = cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0.0)
	return annotated


def open_camera(camera_index: Any, width: int, height: int) -> tuple[cv2.VideoCapture, int, int]:
	"""Open the camera and request the configured resolution."""
	backends: list[Any] = [cv2.CAP_V4L2, cv2.CAP_ANY]
	capture: cv2.VideoCapture | None = None

	for backend in backends:
		capture = cv2.VideoCapture(camera_index, backend)
		if capture.isOpened():
			break
		capture.release()
		capture = None

	if capture is None or not capture.isOpened():
		raise RuntimeError(f"Could not open camera source: {camera_index}")

	capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	time.sleep(0.5)

	actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	return capture, actual_w, actual_h


def save_frame(frame: np.ndarray, prefix: str = "segmentation_check") -> Path:
	"""Save the current annotated frame to the local photos_checked directory."""
	SAVE_DIR.mkdir(parents=True, exist_ok=True)
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	file_path = SAVE_DIR / f"{prefix}_{timestamp}_{int(time.time() * 1000) % 1000:03d}.jpg"
	cv2.imwrite(str(file_path), frame)
	return file_path


def main() -> None:
	"""Run the live segmentation viewer."""
	settings = load_settings()

	if not settings.model_path.exists():
		raise FileNotFoundError(f"Model file not found: {settings.model_path}")

	print("=" * 60)
	print("YOLO SEGMENTATION LIVE VIEW")
	print("=" * 60)
	print(f"Model       : {settings.model_path}")
	print(f"Camera      : {settings.camera_index}")
	print(f"Resolution  : {settings.calib_w}x{settings.calib_h}")
	print(f"Conf / IoU  : {settings.conf_thresh} / {settings.iou_thresh}")
	print(f"Save dir    : {SAVE_DIR}")
	print(f"Interval    : {INFERENCE_INTERVAL_SECONDS}s")
	print(f"Saving      : {AUTO_SAVE_COUNT} frames then exiting")
	print("=" * 60)

	model = YOLO(str(settings.model_path))
	capture, actual_w, actual_h = open_camera(settings.camera_index, settings.calib_w, settings.calib_h)

	if (actual_w, actual_h) != (settings.calib_w, settings.calib_h):
		print(f"[INFO] Camera returned {actual_w}x{actual_h} instead of {settings.calib_w}x{settings.calib_h}.")

	last_inference_time = 0.0
	auto_saved = 0

	try:
		while auto_saved < AUTO_SAVE_COUNT:
			ok, frame = capture.read()
			if not ok:
				print("[WARN] Failed to read a frame from the camera.")
				continue

			now = time.time()
			if (now - last_inference_time) < INFERENCE_INTERVAL_SECONDS:
				continue

			results = model.predict(
				source=frame,
				conf=settings.conf_thresh,
				iou=settings.iou_thresh,
				max_det=settings.max_detections,
				verbose=False,
			)

			result = results[0]
			detections = len(result.boxes) if result.boxes is not None else 0
			annotated = annotate_result(frame, result, getattr(model, "names", None))
			last_inference_time = now

			auto_saved += 1
			saved_path = save_frame(annotated, prefix=f"shot_{auto_saved}")
			print(f"[INFO] Saved {auto_saved}/{AUTO_SAVE_COUNT}: {saved_path}  (detections: {detections})")

	finally:
		capture.release()

	print(f"[INFO] Done. {auto_saved} image(s) saved to {SAVE_DIR}")


if __name__ == "__main__":
	main()
