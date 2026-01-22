# measurement.py
import os
import json
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import traceback
import time
from datetime import datetime

# Import all config constants
from config import *

# -------------------------
# Helpers (calibration / projection)
# -------------------------
def load_json(path):
    """Load and parse a JSON file.

    Parameters:
    - path: Path to the JSON file on disk.

    Returns:
    - Parsed Python object (dict/list) from the JSON contents.
    """
    with open(path, 'r') as f:
        return json.load(f)

def force_camera_resolution(cap, w, h):
    """Force an OpenCV capture device to a target resolution.

    Tries to set width/height; returns the actual resolution. Warns if the
    device cannot match the requested resolution (calibration should match).

    Parameters:
    - cap: cv2.VideoCapture instance.
    - w: requested width in pixels.
    - h: requested height in pixels.

    Returns:
    - (aw, ah): actual width and height reported by the camera.
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) #manual exposure
    # cap.set(cv2.CAP_PROP_EXPOSURE,-6)     #Set exposure (negative values = faster shutter)

     # Allow camera to adjust
    time.sleep(2)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if aw != w or ah != h:
        print(f"Warning: camera returned resolution {aw}x{ah}, expected {w}x{h}. Re-calibrate at this resolution if needed.")
    return aw, ah

def compute_camera_plane(R, t):
    """Compute the fabric plane in camera coordinates from extrinsics.

    The plane normal is the camera's Z axis in world-to-camera rotation `R`,
    and the plane offset `d_c` comes from placing the plane through the origin
    of the world frame translated by `t`.

    Parameters:
    - R: 3x3 rotation matrix (world → camera).
    - t: 3-vector translation (world origin in camera coords).

    Returns:
    - (n_c, d_c): plane normal and plane offset for `n_c·X + d_c = 0` in camera coords.
    """
    n_c = R[:, 2].astype(np.float64)
    d_c = -float(n_c.dot(t))  # Plane passing through world origin transformed by t
    return n_c, d_c

def pixel_to_world_using_camera_plane(u, v, K, dist, R, t, n_c, d_c):
    '''This function converts a 2D pixel coordinate (u, v) from the camera image into a 3D 
    world coordinate by intersecting a ray with the calibrated plane where the fabric lies.'''
    try:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)  # 1×1×2 pixel
        und = cv2.undistortPoints(pts, K, dist, P=None)  # Normalize by intrinsics & distortion
        x_n, y_n = float(und[0,0,0]), float(und[0,0,1])  # Normalized camera ray coords
        d_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)  # Ray direction in camera coords
        denom = float(n_c.dot(d_cam))  # Ray·plane normal — checks for parallelism
        eps = 1e-9
        if abs(denom) < eps:
            return None
        s = -d_c / denom  # Ray-plane intersection scale along direction
        X_cam = s * d_cam  # Intersection point in camera coordinates
        X_world = R.T.dot(X_cam - t)  # Transform camera→world using inverse rotation
        return X_world
    except Exception:
        return None

# -------------------------
# Mask helpers (Ultralytics variants)
# -------------------------
def get_instance_mask_as_bitmap(result, idx, h, w):
    '''
    This function extracts an instance segmentation mask from YOLO results and converts it to a binary bitmap.
    '''
    # Method 1: Direct mask data (dense masks)
    try:
        data = result.masks.data  # Torch tensor or ndarray of masks
        try:
            arr = data[idx].cpu().numpy()
        except Exception:
            arr = np.array(data[idx])
        if arr.ndim == 1 and arr.size == h*w:
            arr = arr.reshape(h, w)  # Flatten → image shape
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)  # Resize to frame
        mask = (arr > 0).astype(np.uint8)  # Threshold into binary mask
        if np.count_nonzero(mask) > 0:
            return mask
    except Exception:
        print("Warning: failed to extract mask via direct data method")
        pass

    # Method 2: Polygon XY coordinates
    try:
        poly = result.masks.xy[idx]  # List of (x,y) points
        poly_pts = np.array(poly, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        if poly_pts.ndim == 2 and poly_pts.shape[0] > 2:
            cv2.fillPoly(mask, [poly_pts], 1)  # Rasterize polygon
            if np.count_nonzero(mask) > 0:
                return mask
    except Exception:
        print("Warning: failed to extract mask via polygon XY method")
        pass

    # Method 3: Polygons attribute
    try:
        polylist = getattr(result.masks, "polygons", None)
        if polylist is not None:
            poly_pts = np.array(polylist[idx], dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            if poly_pts.ndim == 2 and poly_pts.shape[0] > 2:
                cv2.fillPoly(mask, [poly_pts], 1)
                if np.count_nonzero(mask) > 0:
                    return mask
    except Exception:
        print("Warning: failed to extract mask via polygon method")
        pass

    return None

# -------------------------
# Simple 1D k-means (k=2)
# -------------------------
def kmeans_1d_two_clusters(values, max_iters=10):
    '''
    This function implements k-means clustering for 1D data with exactly 2 clusters.
    '''
    if values.size < 2:
        return np.zeros(values.shape[0], dtype=int), (float(values.mean()), float(values.mean()))
    c0 = float(values.min())  # Initialize centers to min/max
    c1 = float(values.max())
    labels = np.zeros(values.shape[0], dtype=int)
    for _ in range(max_iters):
        d0 = np.abs(values - c0)  # Distance to center 0
        d1 = np.abs(values - c1)  # Distance to center 1
        new_labels = (d1 < d0).astype(int)  # Assign to closest center
        if new_labels.sum() == 0 or new_labels.sum() == len(values):
            labels = new_labels
            break
        new_c0 = float(values[new_labels == 0].mean()) if (new_labels == 0).any() else c0
        new_c1 = float(values[new_labels == 1].mean()) if (new_labels == 1).any() else c1
        if new_c0 == c0 and new_c1 == c1:
            labels = new_labels
            break
        c0, c1 = new_c0, new_c1
        labels = new_labels
    return labels, (c0, c1)

# -------------------------
# App class
# -------------------------
class StitchMeasurementApp:
    """Application that detects stitched seams on fabric and measures:

    - Distance from seam to fabric edge in millimeters.
    - Average stitch width in millimeters.

    It uses a calibrated camera (intrinsics/extrinsics) and a YOLO segmentation
    model to project pixel measurements onto the fabric plane in world space.
    """
    def __init__(self,
                 calib_path,
                 extr_path,
                 model_path,
                 camera_index=0,
                 calib_w=640,
                 calib_h=640,
                 frame_buffer=8,
                 min_stitches=MIN_STITCHES,
                 stitch_id=STITCH_CLASS_ID,
                 fabric_id=FABRIC_CLASS_ID):
        """Initialize the measurement app with calibration and model.

        Parameters:
        - calib_path: path to camera intrinsics JSON.
        - extr_path: path to camera extrinsics JSON (rvec/tvec).
        - model_path: path to YOLO segmentation model file.
        - camera_index: OpenCV camera index.
        - calib_w/calib_h: resolution used for calibration and capture.
        - frame_buffer: smoothing buffer size for measurements.
        - min_stitches: minimum count to compute averages.
        - stitch_id/fabric_id: class IDs for YOLO results.
        """
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file missing: {calib_path}") #usuful for standolone 
        calib = load_json(calib_path)
        self.K = np.array(calib["camera_matrix"], dtype=np.float64)  # 3×3 intrinsics
        self.dist = np.array(calib["dist_coeffs"], dtype=np.float64).ravel()  # Distortion coeffs

        if not os.path.exists(extr_path):
            raise FileNotFoundError(f"Extrinsics file missing: {extr_path}")
        extr = load_json(extr_path)
        rvec = np.array(extr["rvec"], dtype=np.float64).reshape(3,1)  # Rotation vector
        tvec = np.array(extr["tvec"], dtype=np.float64).reshape(3,)   # Translation vector
        R_mat, _ = cv2.Rodrigues(rvec)  # Convert rvec → 3×3 rotation matrix
        self.R = R_mat #
        self.t = tvec

        self.n_c, self.d_c = compute_camera_plane(self.R, self.t)  # Plane in camera coords

        self.model = YOLO(model_path)  # Load segmentation model
        #self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # for Windows cv2.CAP_DSHOW
        self.cap= cv2.VideoCapture(camera_index,cv2.CAP_V4L2)  # Open camera
        self.aw, self.ah = force_camera_resolution(self.cap, calib_w, calib_h)  # Enforce resolution

        self.frame_buf_dist = deque(maxlen=frame_buffer)   # Median smoothing buffer for distance
        self.frame_buf_width = deque(maxlen=frame_buffer)  # Median smoothing buffer for width
        self.min_stitches = min_stitches
        self.stitch_id = stitch_id
        self.fabric_id = fabric_id
        self.running = True

        print("StitchMeasurementApp initialized.")
        if LOG_DEBUG:
            print("Plane normal (camera coords):", self.n_c, "d_c:", self.d_c)

    def _combine_masks(self, mask_list, h, w):
        """Combine multiple binary masks into one via bitwise OR.

        Parameters:
        - mask_list: list of H×W uint8 masks (values 0/1).
        - h, w: target height/width to ensure consistent shape.

        Returns:
        - Combined binary mask or None if list is empty.
        """
        if not mask_list:
            return None
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in mask_list:
            if m is not None and m.shape == (h, w):
                combined = cv2.bitwise_or(combined, m.astype(np.uint8))  # OR accumulate
        return combined

    def _fabric_upper_envelope(self, fabric_mask):
        """Trace the upper envelope (topmost fabric pixel) per column.

        Returns an array of length W where each entry is the Y index of the
        first non-zero pixel in that column; -1 if no fabric in the column.
        """
        h, w = fabric_mask.shape
        envelope = np.full((w,), -1, dtype=int)
        has_any = fabric_mask.any(axis=0)              # Column-wise fabric presence
        idx_top = np.argmax(fabric_mask > 0, axis=0)   # First index of True per column
        for x in range(w):
            if has_any[x]:
                envelope[x] = idx_top[x]
            else:
                envelope[x] = -1
        return envelope
    
    def _fabric_lower_envelope(self, fabric_mask):
        """Trace the lower envelope (bottommost fabric pixel) per column."""
        h, w = fabric_mask.shape
        envelope = np.full((w,), -1, dtype=int)
        rev = fabric_mask[::-1, :]           # Flip vertically to find bottom via argmax
        has_any = rev.any(axis=0)
        idx_in_rev = np.argmax(rev > 0, axis=0)
        for x in range(w):
            if has_any[x]:
                envelope[x] = h - 1 - idx_in_rev[x]  # Map back to original Y
            else:
                envelope[x] = -1
        return envelope

    def _fabric_edge_points(self, fabric_mask, max_pts=5000):
        """Extract fabric edge points from contours with optional downsampling.

        Parameters:
        - fabric_mask: binary mask where fabric pixels are 1.
        - max_pts: limit returned points to avoid heavy drawing/computation.

        Returns:
        - N×2 array of (x,y) edge points or None if no contours.
        """
        contours, _ = cv2.findContours((fabric_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        pts = np.vstack(contours).squeeze()  # Concatenate all contour points
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if pts.shape[0] > max_pts:
            step = int(np.ceil(pts.shape[0] / float(max_pts)))  # Uniform downsample
            pts = pts[::step]
        return pts

    def process_frame(self, frame):
        """Process a single BGR frame to compute seam metrics and annotations.

        Steps:
        - Run YOLO segmentation to detect fabric and stitches.
        - Build fabric upper envelope (edge) by column.
        - Cluster stitch rows; select the one closest to the fabric edge.
        - Project pixels to world coordinates and measure distances/widths.
        - Average across stitches and smooth across frames.

        Returns:
        - (annotated_frame, measurements_dict)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # YOLO expects RGB

        try:
            results = self.model.predict(rgb, verbose=False, conf=CONF_THRESH, iou=IOU_THRESH, max_det=MAX_DETECTIONS)
            r = results[0]  # First image result
        except Exception as e:
            print("Model inference error:", e)
            traceback.print_exc()
            return frame.copy(), {
                'edge_distance_mm': None,
                'stitch_width_mm': None,
                'stitch_count': 0,
                'timestamp': datetime.now(),
                'error': 'Model inference failed'
            }

        annotated = frame.copy()  # Drawing canvas
        stitch_masks = []
        stitch_boxes = []
        fabric_masks = []

        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                cls_arr = r.boxes.cls.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
            except Exception:
                try:
                    cls_arr = r.boxes.cls.numpy()
                    boxes = r.boxes.xyxy.numpy()
                except Exception:
                    cls_arr = []
                    boxes = []

            for i, cls_id in enumerate(cls_arr):
                try:
                    cid = int(cls_id)
                except Exception:
                    cid = int(float(cls_id))
                x1, y1, x2, y2 = boxes[i]
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])  # Bounding box ints
                mask = None
                try:
                    mask = get_instance_mask_as_bitmap(r, i, h, w)
                except Exception:
                    mask = None

                if cid == self.stitch_id:
                    stitch_masks.append(mask)
                    stitch_boxes.append((x1i, y1i, x2i, y2i))
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 255, 0), 1)
                elif cid == self.fabric_id:
                    if mask is not None:
                        fabric_masks.append(mask)
                    else:
                        tmp = np.zeros((h, w), dtype=np.uint8)  # Fallback: box → mask
                        cv2.rectangle(tmp, (x1i, y1i), (x2i, y2i), 1, -1)
                        fabric_masks.append(tmp)
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 0, 255), 2)

        if LOG_DEBUG:
            valid_stitch_masks = sum(1 for m in stitch_masks if m is not None and m.sum() > 0)
            valid_fabric_masks = sum(1 for m in fabric_masks if m is not None and m.sum() > 0)
            print(f"Detected: {len(stitch_masks)} stitches ({valid_stitch_masks} valid masks), {len(fabric_masks)} fabric instances ({valid_fabric_masks} valid masks)")

        fabric_mask = self._combine_masks(fabric_masks, h, w)  # Merge multiple fabric instances
        if fabric_mask is None or np.count_nonzero(fabric_mask) == 0:
            cv2.putText(annotated, "Fabric not detected", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return annotated, {
                'edge_distance_mm': None,
                'stitch_width_mm': None,
                'stitch_count': 0,
                'timestamp': datetime.now(),
                'error': 'Fabric not detected'
            }

        envelope = self._fabric_lower_envelope(fabric_mask)  # Lower edge per column

        # draw lower envelope
        pts = []
        for x in range(w):
            y = envelope[x]
            if y >= 0:
                pts.append((x, y))
        if pts:
            step = max(1, int(len(pts) / 1000))
            poly = np.array(pts[::step], dtype=np.int32)  # Downsample polyline for drawing
            cv2.polylines(annotated, [poly], isClosed=False, color=(255,128,0), thickness=2)

        # compute stitch centroids and metadata
        stitch_meta = []
        centroids_y = []
        for idx, mask in enumerate(stitch_masks):
            if mask is not None and mask.sum() > 0:
                M = cv2.moments((mask>0).astype(np.uint8))  # Geometric moments for centroid
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                else:
                    x1i, y1i, x2i, y2i = stitch_boxes[idx]
                    cx = float((x1i + x2i) / 2.0)
                    cy = float((y1i + y2i) / 2.0)
                cols = np.where(np.any(mask>0, axis=0))[0]  # Columns where stitch mask exists
                if cols.size > 0:
                    px_width = float(cols.max() - cols.min())  # Horizontal pixel width
                    left_px = float(cols.min())
                    right_px = float(cols.max())
                else:
                    x1i, y1i, x2i, y2i = stitch_boxes[idx]
                    px_width = float(x2i - x1i)
                    left_px = float(x1i)
                    right_px = float(x2i)
            else:
                x1i, y1i, x2i, y2i = stitch_boxes[idx]
                cx = float((x1i + x2i) / 2.0)
                cy = float((y1i + y2i) / 2.0)
                px_width = float(x2i - x1i)
                left_px = float(x1i)
                right_px = float(x2i)

            stitch_meta.append({
                "mask": mask,
                "bbox": stitch_boxes[idx] if idx < len(stitch_boxes) else None,
                "cx": cx,
                "cy": cy,
                "px_width": px_width,
                "left_px": left_px,
                "right_px": right_px
            })
            centroids_y.append(cy)

        if len(stitch_meta) == 0:
            cv2.putText(annotated, "No stitches detected", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return annotated, {
                'edge_distance_mm': None,
                'stitch_width_mm': None,
                'stitch_count': 0,
                'timestamp': datetime.now(),
                'error': 'No stitches detected'
            }

        # cluster into two Y-groups
        labels = np.zeros(len(centroids_y), dtype=int)
        chosen_label = 0
        if not SKIP_CLUSTER and len(centroids_y) >= 2:
            vals = np.array(centroids_y)
            labels, centers = kmeans_1d_two_clusters(vals)  # k=2 on Y positions
            fabric_validYs = envelope[envelope >= 0]        # Edge positions available
            if fabric_validYs.size > 0:
                fabric_mean_y = float(np.mean(fabric_validYs))
                c0_mean = float(vals[labels == 0].mean()) if (labels==0).any() else 1e9
                c1_mean = float(vals[labels == 1].mean()) if (labels==1).any() else 1e9
                dist0 = abs(c0_mean - fabric_mean_y)
                dist1 = abs(c1_mean - fabric_mean_y)
                chosen_label = 0 if dist0 < dist1 else 1  # Pick cluster closest to fabric edge
            else:
                chosen_label = 0
        else:
            labels = np.zeros(len(centroids_y), dtype=int)
            chosen_label = 0

        selected_indices = [i for i, lab in enumerate(labels) if lab == chosen_label]  # Chosen row

        # filter by proximity to envelope
        final_indices = []
        for i in selected_indices:
            cx = int(round(stitch_meta[i]["cx"]))
            cy = stitch_meta[i]["cy"]
            xs = [int(np.clip(cx + dx, 0, w-1)) for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]  # Neighborhood columns
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]  # Valid envelope ys
            if len(env_vals) == 0:
                continue
            env_y = int(round(float(np.median(env_vals))))  # Robust local edge y
            img_dist = float(cy) - float(env_y)  # Pixel delta (stitch below edge)
            if 0 < img_dist < MAX_PX_DISTANCE:
                final_indices.append(i)

        if LOG_DEBUG:
            print("selected_indices:", selected_indices)
            print("final_indices after envelope filter:", final_indices)

        if len(final_indices) == 0:
            final_indices = selected_indices
            if LOG_DEBUG:
                print("Warning: No stitches within envelope range, using all selected stitches")

        fabric_pts = self._fabric_edge_points(fabric_mask)  # For visualization/drawing
        if fabric_pts is None:
            fabric_pts = np.array([[x, envelope[x]] for x in range(w) if envelope[x] >= 0], dtype=np.int32)

        per_dists = []
        per_widths = []

        for i in final_indices:
            cx = stitch_meta[i]["cx"]
            cy = stitch_meta[i]["cy"]
            cx_int = int(np.clip(int(round(cx)), 0, w-1))
            
            xs = [int(np.clip(cx_int + dx, 0, w-1)) for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]  # Local columns
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]
            if len(env_vals) > 0:
                edge_y = float(np.median(env_vals))  # Local edge y (float)
                
                p_stitch = pixel_to_world_using_camera_plane(float(cx), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                p_edge = pixel_to_world_using_camera_plane(float(cx), float(edge_y), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                
                if p_stitch is not None and p_edge is not None:
                    dist_mm = float(np.linalg.norm(p_stitch - p_edge)) * 1000.0  # meters→mm
                    per_dists.append(dist_mm)
                    
                    cv2.line(annotated, (cx_int, int(round(edge_y))), (int(round(cx)), int(round(cy))), (0,255,0), 1)
                    cv2.circle(annotated, (cx_int, int(round(edge_y))), 2, (255,0,255), -1)

            left_px = stitch_meta[i]["left_px"]
            right_px = stitch_meta[i]["right_px"]
            p_left = pixel_to_world_using_camera_plane(float(left_px), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
            p_right = pixel_to_world_using_camera_plane(float(right_px), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
            if p_left is not None and p_right is not None:
                width_mm = float(np.linalg.norm(p_right - p_left)) * 1000.0  # meters→mm
                per_widths.append(width_mm)
                cv2.circle(annotated, (int(round(left_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.circle(annotated, (int(round(right_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.line(annotated, (int(round(left_px)), int(round(cy))), (int(round(right_px)), int(round(cy))), (200,200,0), 1)
            else:
                p_a = pixel_to_world_using_camera_plane(float(cx), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                p_b = pixel_to_world_using_camera_plane(float(cx+10), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                if p_a is not None and p_b is not None:
                    mm_per_10px = float(np.linalg.norm(p_b - p_a)) * 1000.0  # Local scale estimate
                    width_mm = (stitch_meta[i]["px_width"] / 10.0) * mm_per_10px
                    per_widths.append(width_mm)

            cv2.circle(annotated, (int(round(cx)), int(round(cy))), 4, (0,255,0), -1)
            if per_widths:
                cv2.putText(annotated, f"w:{per_widths[-1]:.1f}mm", (int(round(cx))+6, int(round(cy))+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        n_found = len(per_widths)
        avg_dist = float(np.mean(per_dists)) if len(per_dists) >= self.min_stitches else None  # Per-frame average
        avg_width = float(np.mean(per_widths)) if len(per_widths) >= self.min_stitches else None
        
        if avg_dist is not None:
            self.frame_buf_dist.append(avg_dist)
            smooth_dist = float(np.median(self.frame_buf_dist))  # Robust temporal smoothing
        else:
            smooth_dist = None
            
        if avg_width is not None:
            self.frame_buf_width.append(avg_width)
            smooth_width = float(np.median(self.frame_buf_width))
        else:
            smooth_width = None
        
        if smooth_dist is not None and smooth_width is not None:
            info_text = f"Edge Dist: {smooth_dist:.2f}mm | Avg Width: {smooth_width:.2f}mm (n={n_found})"
        elif smooth_dist is not None:
            info_text = f"Edge Distance: {smooth_dist:.2f}mm (n={n_found})"
        elif smooth_width is not None:
            info_text = f"Avg Width: {smooth_width:.2f}mm (n={n_found})"
        else:
            info_text = f"Insufficient stitches (found {n_found}, need {self.min_stitches})"

        contours_vis, _ = cv2.findContours((fabric_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_vis:
            cv2.drawContours(annotated, contours_vis, -1, (255,128,0), 2)  # Outline fabric

        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        detection_info = f"Stitches: {len(stitch_masks)} | Fabric: {len(fabric_masks)}"
        cv2.putText(annotated, detection_info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Return annotated frame and measurements dictionary
        return annotated, {
            'edge_distance_mm': smooth_dist,
            'stitch_width_mm': smooth_width,
            'stitch_count': n_found,
            'timestamp': datetime.now()
        }

    def get_single_measurement(self):
        """Process one frame and return measurements without the continuous loop.

        Returns:
        - (annotated_frame, measurements_dict) or (None, None) if capture fails.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        annotated, measurements = self.process_frame(frame)
        return annotated, measurements

    def run(self):
        """Continuous capture loop for standalone operation.

        Saves annotated frames periodically, shows live preview, and prints
        measurement summaries. Press 'q' to quit.
        """
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