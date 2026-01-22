# stitch_measurement_with_plane_fixed.py
import os
import json
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import traceback

# -------------------------
# Config / files / camera
# -------------------------
CALIB_FILE = "camera_calibration.json"
EXTRINSICS_FILE = "camera_extrinsics.json"
MODEL_PATH = "yolov8n_seg_200.pt"
CAMERA_INDEX = 1
CALIB_W, CALIB_H = 640, 640

# smoothing
FRAME_BUFFER = 8  # median filter across frames

# measurement behaviour (change if your model uses different IDs)
STITCH_CLASS_ID = 0   # model class id for stitch
FABRIC_CLASS_ID = 1   # model class id for fabric (the cloth)
MIN_STITCHES = 3      # minimum stitches to compute average

# inference tuning
CONF_THRESH = 0.20  # lowered for better detection
IOU_THRESH = 0.45
MAX_DETECTIONS = 200

# debug
LOG_DEBUG = True    # set True to print debug info and counts
SKIP_CLUSTER = False  # if True, don't try to cluster into 2 stitch lines

# robustness/tuning
MAX_EDGE_CANDIDATES = 20  # number of nearest contour points to try per stitch
MAX_PX_DISTANCE = 150     # max pixel distance between stitch centroid and fabric edge (image space)
ENVELOPE_NEIGHBORHOOD = 3 # columns around centroid to average envelope y

# -------------------------
# Helpers (calibration / projection)
# -------------------------
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def force_camera_resolution(cap, w, h):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # actual width
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # actual height
    if aw != w or ah != h:
        print(f"Warning: camera returned resolution {aw}x{ah}, expected {w}x{h}. Re-calibrate at this resolution if needed.")
    return aw, ah

def compute_camera_plane(R, t):
    n_c = R[:, 2].astype(np.float64)
    d_c = -float(n_c.dot(t)) 
    return n_c, d_c

def pixel_to_world_using_camera_plane(u, v, K, dist, R, t, n_c, d_c):

    '''This function converts a 2D pixel coordinate (u, v) from the camera image into a 3D 
    world coordinate by intersecting a ray with the calibrated plane where the fabric lies.'''
    try:
        pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, K, dist, P=None) #Takes the pixel coordinate and removes lens distortion using camera intrinsics 
        x_n, y_n = float(und[0,0,0]), float(und[0,0,1])
        d_cam = np.array([x_n, y_n, 1.0], dtype=np.float64)
        denom = float(n_c.dot(d_cam))
        eps = 1e-9
        if abs(denom) < eps:
            return None
        s = -d_c / denom
        X_cam = s * d_cam
        X_world = R.T.dot(X_cam - t)
        return X_world
    except Exception:
        return None

# -------------------------
# Mask helpers (Ultralytics variants)
# -------------------------
def get_instance_mask_as_bitmap(result, idx, h, w):
    '''
    This function extracts an instance segmentation mask from YOLO results and converts it to a binary bitmap. It tries 3 different methods
    because different versions of Ultralytics YOLO store masks in different formats.
    
    :param result: YOLO result object
    :param idx: index of the detection
    :param h: image height
    :param w: image width
    '''
    # Method 1: Direct mask data (dense masks)
    try:
        data = result.masks.data
        try:
            arr = data[idx].cpu().numpy()
        except Exception:
            arr = np.array(data[idx])
        if arr.ndim == 1 and arr.size == h*w:
            arr = arr.reshape(h, w)
        # Resize if mask dimensions don't match
        if arr.shape != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (arr > 0).astype(np.uint8)
        if np.count_nonzero(mask) > 0:  # validate mask has content
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
        pass

    return None

# -------------------------
# Simple 1D k-means (k=2)
# -------------------------
def kmeans_1d_two_clusters(values, max_iters=10):
    '''
    This function implements k-means clustering for 1D data with exactly 2 clusters.
      It's used to separate stitches into two horizontal rows
      (e.g., top row and bottom row) based on their Y-coordinates.
    
    :param values: values to cluster (1D numpy array)
    :param max_iters: maximum number of iterations for k-means
    '''
    if values.size < 2:
        return np.zeros(values.shape[0], dtype=int), (float(values.mean()), float(values.mean()))
    c0 = float(values.min()) #cluster 0 center = minimum value
    c1 = float(values.max()) #cluster 1 center = maximum value
    labels = np.zeros(values.shape[0], dtype=int)
    for _ in range(max_iters):
        d0 = np.abs(values - c0) #distance to cluster 0 center
        d1 = np.abs(values - c1) #distance to cluster 1 center
        new_labels = (d1 < d0).astype(int) #assign to cluster 1 if closer to c1
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
        R_mat, _ = cv2.Rodrigues(rvec) #Convert rotation vector to rotation matrix
        self.R = R_mat # rotation matrix
        self.t = tvec # translation vector

        self.n_c, self.d_c = compute_camera_plane(self.R, self.t) #setup plane normal and offset
        '''
        A plane in 3D space is defined by: n · X + d = 0
        Where:
        n = plane normal vector (perpendicular to the plane surface)
        X = any point on the plane
        d = signed distance from origin to plane
'''
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) #Open camera with DirectShow backend
        self.aw, self.ah = force_camera_resolution(self.cap, calib_w, calib_h)

        self.frame_buf_dist = deque(maxlen=frame_buffer) # buffer for distance smoothing
        self.frame_buf_width = deque(maxlen=frame_buffer) # buffer for width smoothing
        self.min_stitches = min_stitches
        self.stitch_id = stitch_id
        self.fabric_id = fabric_id
        self.running = True

        print("StitchMeasurementApp initialized.")
        if LOG_DEBUG:
            print("Plane normal (camera coords):", self.n_c, "d_c:", self.d_c)

    def _combine_masks(self, mask_list, h, w):
        '''
        Combines multiple fabric instance masks into a single unified mask.
        '''
        if not mask_list:
            return None
        combined = np.zeros((h, w), dtype=np.uint8)
        for m in mask_list:
            if m is not None and m.shape == (h, w):
                combined = cv2.bitwise_or(combined, m.astype(np.uint8))
        return combined

    def _fabric_upper_envelope(self, fabric_mask):
        '''Purpose: Finds the top-most fabric pixel at each column (x-coordinate)
        This is the fabric edge closest to the camera in angled view.
        '''
        h, w = fabric_mask.shape
        envelope = np.full((w,), -1, dtype=int)
        has_any = fabric_mask.any(axis=0)
        idx_top = np.argmax(fabric_mask > 0, axis=0)
        for x in range(w):
            if has_any[x]:
                envelope[x] = idx_top[x]
            else:
                envelope[x] = -1
        return envelope
    
    def _fabric_lower_envelope(self, fabric_mask):
        '''Purpose: Finds the bottom-most fabric pixel at each column (x-coordinate)
        '''
        h, w = fabric_mask.shape
        envelope = np.full((w,), -1, dtype=int)
        rev = fabric_mask[::-1, :]
        has_any = rev.any(axis=0)
        idx_in_rev = np.argmax(rev > 0, axis=0)
        for x in range(w):
            if has_any[x]:
                envelope[x] = h - 1 - idx_in_rev[x]
            else:
                envelope[x] = -1
        return envelope

    def _fabric_edge_points(self, fabric_mask, max_pts=5000):
        '''Purpose: Extracts all points along the outer boundary of the fabric mask.'''
        contours, _ = cv2.findContours((fabric_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        pts = np.vstack(contours).squeeze()
        if pts.ndim == 1: # single point case
            pts = pts.reshape(-1, 2) # ensure 2D array
        if pts.shape[0] > max_pts: #
            step = int(np.ceil(pts.shape[0] / float(max_pts)))
            pts = pts[::step]
        return pts

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self.model.predict(rgb, verbose=False, conf=CONF_THRESH, iou=IOU_THRESH, max_det=MAX_DETECTIONS)
            r = results[0]
        except Exception as e:
            print("Model inference error:", e)
            traceback.print_exc()
            return frame.copy(), "Model error"

        annotated = frame.copy()
        stitch_masks = []
        stitch_boxes = []
        fabric_masks = []

        if hasattr(r, "boxes") and r.boxes is not None:# check if boxes exist
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
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                mask = None
                try:
                    mask = get_instance_mask_as_bitmap(r, i, h, w)
                except Exception:
                    mask = None

                if cid == self.stitch_id:
                    stitch_masks.append(mask)
                    stitch_boxes.append((x1i, y1i, x2i, y2i))
                    # Draw stitch bounding boxes in cyan for debugging
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 255, 0), 1)
                elif cid == self.fabric_id:
                    if mask is not None:
                        fabric_masks.append(mask)
                    else:
                        tmp = np.zeros((h, w), dtype=np.uint8)
                        cv2.rectangle(tmp, (x1i, y1i), (x2i, y2i), 1, -1)
                        fabric_masks.append(tmp)
                    # Draw fabric bounding boxes in magenta for debugging
                    cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (255, 0, 255), 2)

        if LOG_DEBUG:
            # Count valid masks (not None)
            valid_stitch_masks = sum(1 for m in stitch_masks if m is not None and m.sum() > 0)
            valid_fabric_masks = sum(1 for m in fabric_masks if m is not None and m.sum() > 0)
            print(f"Detected: {len(stitch_masks)} stitches ({valid_stitch_masks} valid masks), {len(fabric_masks)} fabric instances ({valid_fabric_masks} valid masks)")

        fabric_mask = self._combine_masks(fabric_masks, h, w)
        if fabric_mask is None or np.count_nonzero(fabric_mask) == 0:
            cv2.putText(annotated, "Fabric not detected", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return annotated, "Fabric not detected"

        envelope = self._fabric_upper_envelope(fabric_mask)  # Use top edge (closest to camera)

        # draw upper envelope (fabric edge)
        pts = []
        for x in range(w):
            y = envelope[x]
            if y >= 0:
                pts.append((x, y))
        if pts:
            step = max(1, int(len(pts) / 1000))
            poly = np.array(pts[::step], dtype=np.int32)
            cv2.polylines(annotated, [poly], isClosed=False, color=(255,128,0), thickness=2)

        # compute stitch centroids and metadata
        stitch_meta = []
        centroids_y = []
        for idx, mask in enumerate(stitch_masks):
            if mask is not None and mask.sum() > 0:
                M = cv2.moments((mask>0).astype(np.uint8))
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                else:
                    x1i, y1i, x2i, y2i = stitch_boxes[idx]
                    cx = float((x1i + x2i) / 2.0)
                    cy = float((y1i + y2i) / 2.0)
                cols = np.where(np.any(mask>0, axis=0))[0]
                if cols.size > 0:
                    px_width = float(cols.max() - cols.min())
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
            return annotated, "No stitches detected"

        # cluster into two Y-groups if enough stitches and not skipped
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
                dist0 = abs(c0_mean - fabric_mean_y)
                dist1 = abs(c1_mean - fabric_mean_y)
                chosen_label = 0 if dist0 < dist1 else 1
            else:
                chosen_label = 0
        else:
            labels = np.zeros(len(centroids_y), dtype=int)
            chosen_label = 0

        # select stitches in chosen cluster
        selected_indices = [i for i, lab in enumerate(labels) if lab == chosen_label]

        # filter by proximity to envelope - stitches should be below the top edge
        final_indices = []
        for i in selected_indices:
            cx = int(round(stitch_meta[i]["cx"]))
            cy = stitch_meta[i]["cy"]
            xs = [int(np.clip(cx + dx, 0, w-1)) for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]
            if len(env_vals) == 0:
                continue
            env_y = int(round(float(np.median(env_vals))))
            # Stitches should be below (greater y) the top fabric edge
            img_dist = float(cy) - float(env_y)  # positive if stitch is below edge
            if 0 < img_dist < MAX_PX_DISTANCE:  # within 150px below the top edge
                final_indices.append(i)

        if LOG_DEBUG:
            print("selected_indices:", selected_indices)
            print("final_indices after envelope filter:", final_indices)

        # If no stitches pass filter, use all selected stitches
        if len(final_indices) == 0:
            final_indices = selected_indices
            if LOG_DEBUG:
                print("Warning: No stitches within envelope range, using all selected stitches")

        # prepare fabric contour points for nearest search (if needed)
        fabric_pts = self._fabric_edge_points(fabric_mask)
        if fabric_pts is None:
            # fallback: use envelope sampling as points
            fabric_pts = np.array([[x, envelope[x]] for x in range(w) if envelope[x] >= 0], dtype=np.int32)

        per_dists = []  # distance from stitch centroid to fabric top edge
        per_widths = []  # stitch widths from bounding boxes

        for i in final_indices:
            cx = stitch_meta[i]["cx"]
            cy = stitch_meta[i]["cy"]
            cx_int = int(np.clip(int(round(cx)), 0, w-1))
            
            # Get fabric edge at this stitch position
            xs = [int(np.clip(cx_int + dx, 0, w-1)) for dx in range(-ENVELOPE_NEIGHBORHOOD, ENVELOPE_NEIGHBORHOOD+1)]
            env_vals = [envelope[x] for x in xs if envelope[x] >= 0]
            if len(env_vals) > 0:
                edge_y = float(np.median(env_vals))
                
                # Calculate distance from centroid to fabric edge (3D projection)
                p_stitch = pixel_to_world_using_camera_plane(float(cx), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                p_edge = pixel_to_world_using_camera_plane(float(cx), float(edge_y), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                
                if p_stitch is not None and p_edge is not None:
                    dist_mm = float(np.linalg.norm(p_stitch - p_edge)) * 1000.0
                    per_dists.append(dist_mm)
                    
                    # Draw distance line from edge to centroid
                    cv2.line(annotated, (cx_int, int(round(edge_y))), (int(round(cx)), int(round(cy))), (0,255,0), 1)
                    cv2.circle(annotated, (cx_int, int(round(edge_y))), 2, (255,0,255), -1)  # edge point

            # compute width in mm using bounding box
            left_px = stitch_meta[i]["left_px"]
            right_px = stitch_meta[i]["right_px"]
            p_left = pixel_to_world_using_camera_plane(float(left_px), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
            p_right = pixel_to_world_using_camera_plane(float(right_px), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
            if p_left is not None and p_right is not None:
                width_mm = float(np.linalg.norm(p_right - p_left)) * 1000.0
                per_widths.append(width_mm)
                # draw width endpoints
                cv2.circle(annotated, (int(round(left_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.circle(annotated, (int(round(right_px)), int(round(cy))), 3, (200,200,0), -1)
                cv2.line(annotated, (int(round(left_px)), int(round(cy))), (int(round(right_px)), int(round(cy))), (200,200,0), 1)
            else:
                # try local pixel-to-mm estimate
                p_a = pixel_to_world_using_camera_plane(float(cx), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                p_b = pixel_to_world_using_camera_plane(float(cx+10), float(cy), self.K, self.dist, self.R, self.t, self.n_c, self.d_c)
                if p_a is not None and p_b is not None:
                    mm_per_10px = float(np.linalg.norm(p_b - p_a)) * 1000.0
                    width_mm = (stitch_meta[i]["px_width"] / 10.0) * mm_per_10px
                    per_widths.append(width_mm)

            # annotate centroid
            cv2.circle(annotated, (int(round(cx)), int(round(cy))), 4, (0,255,0), -1)
            if per_widths:
                cv2.putText(annotated, f"w:{per_widths[-1]:.1f}mm", (int(round(cx))+6, int(round(cy))+6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # finalize averages and display
        n_found = len(per_widths)
        avg_dist = float(np.mean(per_dists)) if len(per_dists) >= self.min_stitches else None
        avg_width = float(np.mean(per_widths)) if len(per_widths) >= self.min_stitches else None
        
        # Apply temporal smoothing
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
        
        # Build info text
        if smooth_dist is not None and smooth_width is not None:
            info_text = f"Edge Dist: {smooth_dist:.2f}mm | Avg Width: {smooth_width:.2f}mm (n={n_found})"
        elif smooth_dist is not None:
            info_text = f"Edge Distance: {smooth_dist:.2f}mm (n={n_found})"
        elif smooth_width is not None:
            info_text = f"Avg Width: {smooth_width:.2f}mm (n={n_found})"
        else:
            info_text = f"Insufficient stitches (found {n_found}, need {self.min_stitches})"

        # draw fabric contour for clarity
        contours_vis, _ = cv2.findContours((fabric_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_vis:
            cv2.drawContours(annotated, contours_vis, -1, (255,128,0), 2)

        cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # Add detection count overlay for debugging
        detection_info = f"Stitches: {len(stitch_masks)} | Fabric: {len(fabric_masks)}"
        cv2.putText(annotated, detection_info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated, info_text

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("No frame, retrying...")
                continue
            annotated, info = self.process_frame(frame)
            cv2.imshow("Stitch -> Fabric-edge distances & widths", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

# -------------------------
# Entry
# -------------------------

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(base_dir, CALIB_FILE)
    extr_path = os.path.join(base_dir, EXTRINSICS_FILE)

    app = StitchMeasurementApp(calib_path, extr_path, MODEL_PATH, camera_index=CAMERA_INDEX,
                               calib_w=CALIB_W, calib_h=CALIB_H, frame_buffer=FRAME_BUFFER,
                               min_stitches=MIN_STITCHES, stitch_id=STITCH_CLASS_ID, fabric_id=FABRIC_CLASS_ID)
    app.run()
