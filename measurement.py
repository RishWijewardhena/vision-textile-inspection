# measurement.py
import os
import json
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
from datetime import datetime

from config import *

# -------------------------
# Helper Functions
# -------------------------
def load_json(path):
    """Load and parse a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def force_camera_resolution(cap, w, h):
    """Set camera resolution and verify."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    time.sleep(2)  # Allow camera to adjust
    
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Disable auto exposure (V4L2 expects 1 = manual, 3 = auto)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_EXPOSURE)  # Adjust this value``

    # # Optional: set gain
    # cap.set(cv2.CAP_PROP_GAIN, 0)

    if aw != w or ah != h:
        print(f"Warning: camera resolution {aw}x{ah}, expected {w}x{h}")
    return aw, ah

def compute_camera_plane(R, t):
    """Compute fabric plane in camera coordinates from extrinsics."""
    n_c = R[:, 2].astype(np.float64)
    d_c = -float(n_c.dot(t))
    return n_c, d_c

def pixel_to_world_using_camera_plane(u, v, K, dist, R, t, n_c, d_c):
    """Convert 2D pixel to 3D world coordinate via ray-plane intersection."""
    try:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=None)
        x_n, y_n = float(und[0,0,0]), float(und[0,0,1])
        d_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        denom = float(n_c.dot(d_cam))
        if abs(denom) < 1e-9:
            return None
        s = -d_c / denom
        X_cam = s * d_cam
        X_world = R.T.dot(X_cam - t)
        return X_world
    except Exception:
        return None

# -------------------------
# Mask Extraction
# -------------------------
def get_instance_mask_as_bitmap(result, idx, h, w):
    """Extract instance segmentation mask from YOLO results as binary bitmap."""
    # Method 1: Direct mask data
    try:
        data = result.masks.data
        arr = data[idx].cpu().numpy() if hasattr(data[idx], 'cpu') else np.array(data[idx])
        if arr.ndim == 1 and arr.size == h*w:
            arr = arr.reshape(h, w)
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (arr > 0).astype(np.uint8)
        if np.count_nonzero(mask) > 0:
            return mask
    except Exception:
        pass

    # Method 2: Polygon XY coordinates
    try:
        poly = result.masks.xy[idx]
        poly_pts = np.array(poly, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if poly_pts.ndim == 2 and poly_pts.shape[0] > 2:
            cv2.fillPoly(mask, [poly_pts], 1)
            if np.count_nonzero(mask) > 0:
                return mask
    except Exception:
        pass

    return None

def kmeans_1d_two_clusters(values, max_iters=10):
    """Simple 1D k-means clustering with k=2."""
    if values.size < 2:
        return np.zeros(values.shape[0], dtype=int), (float(values.mean()), float(values.mean()))
    
    c0, c1 = float(values.min()), float(values.max())
    labels = np.zeros(values.shape[0], dtype=int)
    
    for _ in range(max_iters):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        new_labels = (d1 < d0).astype(int)
        
        if new_labels.sum() == 0 or new_labels.sum() == len(values):
            break
            
        new_c0 = float(values[new_labels == 0].mean()) if (new_labels == 0).any() else c0
        new_c1 = float(values[new_labels == 1].mean()) if (new_labels == 1).any() else c1
        
        if new_c0 == c0 and new_c1 == c1:
            break
            
        c0, c1 = new_c0, new_c1
        labels = new_labels
        
    return labels, (c0, c1)

# -------------------------
# Stitch Measurement Application
# -------------------------
class StitchMeasurementApp:
    """Detects fabric seams and measures distance from seam to fabric edge."""
    
    def __init__(self, calib_path, extr_path, model_path, camera_index=0,
                 calib_w=640, calib_h=640, frame_buffer=8,
                 min_stitches=MIN_STITCHES, stitch_id=STITCH_CLASS_ID,
                 fabric_id=FABRIC_CLASS_ID):
        """Initialize the measurement app with calibration and model."""
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file missing: {calib_path}")
        calib = load_json(calib_path)
        self.K = np.array(calib["camera_matrix"], dtype=np.float64)
        self.dist = np.array(calib["dist_coeffs"], dtype=np.float64).ravel()

        if not os.path.exists(extr_path):
            raise FileNotFoundError(f"Extrinsics file missing: {extr_path}")
        extr = load_json(extr_path)
        rvec = np.array(extr["rvec"], dtype=np.float64).reshape(3,1)
        tvec = np.array(extr["tvec"], dtype=np.float64).reshape(3,)
        R_mat, _ = cv2.Rodrigues(rvec)
        self.R = R_mat
        self.t = tvec

        self.n_c, self.d_c = compute_camera_plane(self.R, self.t)

        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.aw, self.ah = force_camera_resolution(self.cap, calib_w, calib_h)

        self.frame_buf_dist = deque(maxlen=frame_buffer)
        self.frame_buf_width = deque(maxlen=frame_buffer)
        self.min_stitches = min_stitches
        self.stitch_id = stitch_id
        self.fabric_id = fabric_id
        self.running = True

        print("StitchMeasurementApp initialized.")
        if LOG_DEBUG:
            print("Plane normal (camera coords):", self.n_c, "d_c:", self.d_c)

    def _combine_masks(self, mask_list, h, w):
        """Combine multiple binary masks via bitwise OR."""
        if not mask_list:
            return None
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in mask_list:
            if m is not None and m.shape == (h, w):
                combined = cv2.bitwise_or(combined, m.astype(np.uint8))
        return combined

    def _fabric_lower_envelope(self, fabric_mask):
        """Trace the lower envelope (bottommost fabric pixel) per column."""
        h, w = fabric_mask.shape
        envelope = np.full((w,), -1, dtype=int)
        rev = fabric_mask[::-1, :]
        has_any = rev.any(axis=0)
        idx_in_rev = np.argmax(rev > 0, axis=0)
        for x in range(w):
            if has_any[x]:
                envelope[x] = h - 1 - idx_in_rev[x]
        return envelope

    def process_frame(self, frame):
        """Process frame to compute seam metrics and annotations."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self.model.predict(rgb, verbose=False, conf=CONF_THRESH, 
                                        iou=IOU_THRESH, max_det=MAX_DETECTIONS)
            r = results[0]
        except Exception as e:
            print("Model inference error:", e)
            return frame.copy(), {'edge_distance_mm': None, 'stitch_width_mm': None,
                                 'stitch_count': 0, 'timestamp': datetime.now(),
                                 'error': 'Model inference failed'}

        annotated = frame.copy()
        stitch_masks, stitch_boxes, fabric_masks = [], [], []

        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                cls_arr = r.boxes.cls.cpu().numpy() if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.numpy()
                boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy.numpy()
            except Exception:
                cls_arr, boxes = [], []

            for i, cls_id in enumerate(cls_arr):
                cid = int(cls_id)
                x1, y1, x2, y2 = map(int, boxes[i])
                mask = get_instance_mask_as_bitmap(r, i, h, w)

                if cid == self.stitch_id:
                    stitch_masks.append(mask)
                    stitch_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)
                elif cid == self.fabric_id:
                    if mask is not None:
                        fabric_masks.append(mask)
                    else:
                        tmp = np.zeros((h, w), dtype=np.uint8)
                        cv2.rectangle(tmp, (x1, y1), (x2, y2), 1, -1)
                        fabric_masks.append(tmp)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)

        if LOG_DEBUG:
            valid_stitches = sum(1 for m in stitch_masks if m is not None and m.sum() > 0)
            valid_fabrics = sum(1 for m in fabric_masks if m is not None and m.sum() > 0)
            print(f"Detected: {len(stitch_masks)} stitches ({valid_stitches} valid), "
                  f"{len(fabric_masks)} fabrics ({valid_fabrics} valid)")

        fabric_mask = self._combine_masks(fabric_masks, h, w)
        if fabric_mask is None or np.count_nonzero(fabric_mask) == 0:
            cv2.putText(annotated, "Fabric not detected", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return annotated, {'edge_distance_mm': None, 'stitch_width_mm': None,
                              'stitch_count': 0, 'timestamp': datetime.now(),
                              'error': 'Fabric not detected'}

        envelope = self._fabric_lower_envelope(fabric_mask)

        # Draw envelope
        pts = [(x, envelope[x]) for x in range(w) if envelope[x] >= 0]
        if pts:
            step = max(1, len(pts) // 1000)
            poly = np.array(pts[::step], dtype=np.int32)
            cv2.polylines(annotated, [poly], False, (255,128,0), 2)

        # Compute stitch centroids
        stitch_meta = []
        centroids_y = []
        for idx, mask in enumerate(stitch_masks):
            if mask is not None and mask.sum() > 0:
                M = cv2.moments((mask>0).astype(np.uint8))
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                else:
                    x1, y1, x2, y2 = stitch_boxes[idx]
                    cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                cols = np.where(np.any(mask>0, axis=0))[0]
                if cols.size > 0:
                    px_width = float(cols.max() - cols.min())
                    left_px, right_px = float(cols.min()), float(cols.max())
                else:
                    x1, y1, x2, y2 = stitch_boxes[idx]
                    px_width = float(x2 - x1)
                    left_px, right_px = float(x1), float(x2)
            else:
                x1, y1, x2, y2 = stitch_boxes[idx]
                cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                px_width = float(x2 - x1)
                left_px, right_px = float(x1), float(x2)

            stitch_meta.append({
                "mask": mask, "bbox": stitch_boxes[idx] if idx < len(stitch_boxes) else None,
                "cx": cx, "cy": cy, "px_width": px_width,
                "left_px": left_px, "right_px": right_px
            })
            centroids_y.append(cy)

        if len(stitch_meta) == 0:
            cv2.putText(annotated, "No stitches detected", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return annotated, {'edge_distance_mm': None, 'stitch_width_mm': None,
                              'stitch_count': 0, 'timestamp': datetime.now(),
                              'error': 'No stitches detected'}

        # Cluster stitches and select closest to fabric edge
        labels = np.zeros(len(centroids_y), dtype=int)
        chosen_label = 0
        if not SKIP_CLUSTER and len(centroids_y) >= 2:
            vals = np.array(centroids_y)
            labels, centers = kmeans_1d_two_clusters(vals)
            fabric_validYs = envelope[envelope >= 0]
            if fabric_validYs.size > 0:
                fabric_mean_y = float(np.mean(fabric_validYs))
                c0_mean = float(vals[labels == 0].mean()) if (labels==0).any() else 1e9
                c1_mean = float(vals[labels == 1].mean()) if (labels==1).any() else 1e9
                chosen_label = 0 if abs(c0_mean - fabric_mean_y) < abs(c1_mean - fabric_mean_y) else 1

        selected_indices = [i for i, lab in enumerate(labels) if lab == chosen_label]

        # Filter by proximity to envelope
        final_indices = []
        for i in selected_indices:
            cx = int(round(stitch_meta[i]["cx"]))
            cy = stitch_meta[i]["cy"]
            xs = [int(np.clip(cx + dx, 0, w-1)) 
                  for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]
            if len(env_vals) == 0:
                continue
            env_y = int(round(float(np.median(env_vals))))
            img_dist = abs(float(cy) - float(env_y))
            if img_dist < MAX_PX_DISTANCE:
                final_indices.append(i)

        if LOG_DEBUG:
            print("selected_indices:", selected_indices)
            print("final_indices:", final_indices)

        if len(final_indices) == 0:
            final_indices = selected_indices
            if LOG_DEBUG:
                print("Warning: No stitches within envelope range, using all selected")

        # Measure distances and widths
        per_dists, per_widths = [], []

        for i in final_indices:
            cx, cy = stitch_meta[i]["cx"], stitch_meta[i]["cy"]
            cx_int = int(np.clip(int(round(cx)), 0, w-1))
            
            xs = [int(np.clip(cx_int + dx, 0, w-1)) 
                  for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]
            
            if len(env_vals) > 0:
                edge_y = float(np.median(env_vals))
                p_stitch = pixel_to_world_using_camera_plane(float(cx), float(cy), 
                                                             self.K, self.dist, self.R, 
                                                             self.t, self.n_c, self.d_c)
                p_edge = pixel_to_world_using_camera_plane(float(cx), float(edge_y), 
                                                           self.K, self.dist, self.R, 
                                                           self.t, self.n_c, self.d_c)
                
                if p_stitch is not None and p_edge is not None:
                    dist_mm = float(np.linalg.norm(p_stitch - p_edge)) * 1000.0
                    per_dists.append(dist_mm)
                    cv2.line(annotated, (cx_int, int(round(edge_y))), 
                            (int(round(cx)), int(round(cy))), (0,255,0), 1)
                    cv2.circle(annotated, (cx_int, int(round(edge_y))), 2, (255,0,255), -1)

            left_px, right_px = stitch_meta[i]["left_px"], stitch_meta[i]["right_px"]
            p_left = pixel_to_world_using_camera_plane(float(left_px), float(cy), 
                                                       self.K, self.dist, self.R, 
                                                       self.t, self.n_c, self.d_c)
            p_right = pixel_to_world_using_camera_plane(float(right_px), float(cy), 
                                                        self.K, self.dist, self.R, 
                                                        self.t, self.n_c, self.d_c)
            
            if p_left is not None and p_right is not None:
                width_mm = float(np.linalg.norm(p_right - p_left)) * 1000.0
                per_widths.append(width_mm)
                cv2.circle(annotated, (int(round(left_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.circle(annotated, (int(round(right_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.line(annotated, (int(round(left_px)), int(round(cy))), 
                        (int(round(right_px)), int(round(cy))), (200,200,0), 1)

            cv2.circle(annotated, (int(round(cx)), int(round(cy))), 10, (255,0,0), -1)
            if per_widths:
                cv2.putText(annotated, f"w:{per_widths[-1]:.1f}mm", 
                           (int(round(cx))+6, int(round(cy))+6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        n_found = len(per_widths)
        avg_dist = float(np.mean(per_dists)) if len(per_dists) >= self.min_stitches else None
        avg_width = float(np.mean(per_widths)) if len(per_widths) >= self.min_stitches else None
        
        if avg_dist is not None:
            self.frame_buf_dist.append(avg_dist)
            smooth_dist = float(np.median(self.frame_buf_dist))
        else:
            smooth_dist = None
            
        if avg_width is not None:
            self.frame_buf_width.append(avg_width)
            smooth_width = float(np.median(self.frame_buf_width))
        else:
            smooth_width = None
        
        # Display info
        if smooth_dist is not None and smooth_width is not None:
            info_text = f"Edge Dist: {smooth_dist:.2f}mm | Avg Width: {smooth_width:.2f}mm (n={n_found})"
        elif smooth_dist is not None:
            info_text = f"Edge Distance: {smooth_dist:.2f}mm (n={n_found})"
        elif smooth_width is not None:
            info_text = f"Avg Width: {smooth_width:.2f}mm (n={n_found})"
        else:
            info_text = f"Insufficient stitches (found {n_found}, need {self.min_stitches})"

        contours_vis, _ = cv2.findContours((fabric_mask>0).astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_vis:
            cv2.drawContours(annotated, contours_vis, -1, (0,0,255), 2)

        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        detection_info = f"Stitches: {len(stitch_masks)} | Fabric: {len(fabric_masks)}"
        cv2.putText(annotated, detection_info, (10, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated, {
            'edge_distance_mm': smooth_dist,
            'stitch_width_mm': smooth_width,
            'stitch_count': n_found,
            'timestamp': datetime.now()
        }

    def get_single_measurement(self):
        """Process one frame and return measurements."""
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        return self.process_frame(frame)

    def run(self):
        """Continuous capture loop for standalone operation."""
        last_inference_time = 0
        frame_count = 0
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Saving annotated images to: {os.path.abspath(SAVE_DIR)}")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("No frame, retrying...")
                continue
            
            current_time = time.time()
            
            if current_time - last_inference_time >= INFERENCE_INTERVAL:
                annotated, measurements = self.process_frame(frame)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"frame_{frame_count:05d}_{timestamp}.jpg")
                cv2.imwrite(save_path, annotated)
                
                info = f"Edge: {measurements.get('edge_distance_mm', 'N/A')}mm | Width: {measurements.get('stitch_width_mm', 'N/A')}mm"
                print(f"Saved: {save_path} | {info}")
                
                cv2.imshow("Stitch Measurement", annotated)
                last_inference_time = current_time
                frame_count += 1
            else:
                cv2.imshow("Stitch Measurement", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\nTotal frames saved: {frame_count}")


# Entry point for standalone testing
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(base_dir, INTRINSICS_FILE)
    extr_path = os.path.join(base_dir, EXTRINSICS_FILE)

    app = StitchMeasurementApp(calib_path, extr_path, MODEL_PATH, camera_index=CAMERA_INDEX,
                               calib_w=CALIB_W, calib_h=CALIB_H, frame_buffer=FRAME_BUFFER,
                               min_stitches=MIN_STITCHES, stitch_id=STITCH_CLASS_ID, fabric_id=FABRIC_CLASS_ID)
    app.run()