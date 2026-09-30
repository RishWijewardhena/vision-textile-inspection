# Vision Textile Inspection

A computer vision stitch measurement system for textile production lines. The
application uses a YOLOv8 segmentation model, calibrated camera geometry, ESP32
stitch counts, MySQL storage, and MQTT status/reset messages to measure seam
allowance, stitch width, and total stitched distance.

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

`main.py` is the runtime orchestrator. It opens the camera, runs the measurement
pipeline from `measurement.py`, reads stitch counts from the ESP32 through
`serial_reader.py`, inserts measurements into the configured database, saves
annotated frames, publishes MQTT heartbeat messages, and listens for MQTT reset
commands.

The current measurement flow is:

1. Detect stitch and fabric masks with `single_needle_model.pt`.
2. Ignore detections outside the configured ROI when ROI filtering is enabled.
3. Convert stitch/fabric pixel positions to world measurements using
   `camera_calibration.json` and `camera_extrinsics.json`.
4. Compute per-stitch seam allowance and stitch width.
5. Apply MAD-based outlier filtering before averaging measurements.
6. Smooth accepted values with short rolling buffers.
7. Validate values against configured seam/stitch limits.
8. Use `CONFIRM_CONSECUTIVE` to accept sustained high measurements that would
   otherwise be treated as outliers.
9. Combine the latest stitch count delta with stitch width to update total
   distance and write a database row.

## Key Features

- YOLOv8 segmentation for stitch and fabric detection.
- ChArUco camera calibration and extrinsic calibration support.
- Pixel-to-world measurement using camera intrinsics/extrinsics.
- ROI filtering for ignoring detections outside the measurement area.
- MAD filter for robust per-frame outlier rejection.
- `CONFIRM_CONSECUTIVE` confirmation for sustained high seam/stitch readings.
- Rolling measurement buffers for smoother runtime output.
- ESP32 serial integration for stitch counts and reset commands.
- MySQL measurement logging.
- MQTT heartbeat and reset command handling.
- Needle angle monitoring that detects a rotated camera, confirmed by re-checks
  before any alert is sent.
- Retained MQTT camera calibration status (`invalid` / `valid`) and a remote
  command to clear it.
- Annotated image saving under session-specific folders.
- Automatic cleanup of old saved annotations.
- Camera reconnect handling with webcam driver reload.

## Installation

### Prerequisites

- Python 3.11 or higher
- pip
- A Linux camera setup supported by OpenCV/V4L2
- ESP32 or compatible serial stitch counter
- MySQL database/table configured through `.env`

### Setup

```bash
git clone https://github.com/RishWijewardhena/vision-textile-inspection.git
cd vision-textile-inspection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Before running the main app, install the sudoers rule that allows the camera
reload helper to run `modprobe` without a password:

```bash
cd scripts
sudo bash create_sudoers_thread_modprobe.sh
```

This creates the sudoers drop-in used by `main.py` when it reloads the webcam
driver after a disconnect.

Create a `.env` file with the required database settings:

```bash
DB_HOST=your-db-host
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_DATABASE=your-db-name
DB_TABLE=your-machine-or-table-id
```

Optional environment overrides:

```bash
SERIAL_PORT=/dev/ttyACM0
MQTT_SERVER=mqtt.anc.idea8.cloud
MQTT_PORT=8883
MQTT_USERNAME=backend
MQTT_PASSWORD=your-password
MQTT_TLS_INSECURE=true

ROI_ENABLED=true
ROI_X_MIN=100
ROI_X_MAX=1270
ROI_Y_MIN=300
ROI_Y_MAX=760

OUTLIER_MIN_SAMPLES=4
OUTLIER_MAD_SCALE=1.4

SEAM_LENGTH_OFFSET=-1.3
STITCH_WIDTH_OFFSET=-1.0
```

## Usage

Run the main system:

```bash
python main.py
```

Run calibration manually:

```bash
python calibration.py
```

Run the MQTT reset test utility:

```bash
python Utils/mqtt_reset_test.py
```

Run the automated startup/update script:

```bash
bash auto_run.sh
```

During runtime, press `q` in the OpenCV window to quit when `SHOW_WINDOWS` is
enabled. If windows are disabled, stop the process with `Ctrl+C`.

## Measurement Filtering

### MAD Filter

The measurement pipeline uses a Median Absolute Deviation filter in
`measurement.filtered_mean()` to reject per-frame outliers before calculating
the seam allowance and stitch width averages.

The relevant configuration values are in `config.py`:

```python
OUTLIER_MIN_SAMPLES = _env_int("OUTLIER_MIN_SAMPLES", 4)
OUTLIER_MAD_SCALE = _env_float("OUTLIER_MAD_SCALE", 1.4)
```

Behavior:

- If fewer than `OUTLIER_MIN_SAMPLES` values are available, all values are kept.
- If the MAD is near zero, all values are kept.
- Otherwise, robust z-scores are calculated and values above
  `OUTLIER_MAD_SCALE` are removed.
- If filtering removes every value, the raw mean is used as a safe fallback.
- Debug logs print inlier counts as `MAD filter -> seam inliers: x/y, stitch
  inliers: x/y`.

### Runtime Validation and `CONFIRM_CONSECUTIVE`

After offsets and smoothing, `main.py` validates the live measurements against
configured bounds:

```python
Seam_lower_limit = 3.5
Seam_upper_limit = 7.5
stitch_lower_limit = 2.5
stitch_upper_limit = 4.5

CONFIRM_CONSECUTIVE = 3
CONFIRM_TOLERANCE_MM = 0.55
```

Normal valid ranges:

- Seam allowance must be between `Seam_lower_limit` and `Seam_upper_limit`.
- Stitch width must be between `stitch_lower_limit` and `stitch_upper_limit`.

High readings are handled specially. If seam allowance or stitch width is above
the upper limit, the system checks the most recent `CONFIRM_CONSECUTIVE` raw
post-offset samples. If all of those samples remain above
`upper_limit - CONFIRM_TOLERANCE_MM`, the value is accepted as a real sustained
measurement instead of being treated as a false positive.

Low readings below the lower limits are not confirmed this way; they are ignored
and the system falls back to buffered averages when available.

When a confirmed high value is accepted, the rolling valid-measurement buffers
are cleared so the system adapts quickly to the new sustained value.

## Reset Behavior

The MQTT reset topic is based on `DB_TABLE`:

```text
machine/<DB_TABLE>/commands/reset
```

Publishing `reset` to that topic makes the main loop:

- Insert a `(0, 0, 0)` reset row into the database.
- Send `R` to the ESP32 over serial.
- Reset runtime distance and stitch-delta state.
- Reinitialize valid measurement buffers.
- Publish `reset_success` on the same MQTT topic when database and serial reset
  both succeed.

Heartbeat messages are published to:

```text
machine/<DB_TABLE>/status/heartbeat
```

## Camera Rotation and Calibration Status

A background `NeedleAngleWorker` (`needle_angle_measure.py`) checks the needle
angle every `NEEDLE_ANGLE_CHECK_INTERVAL` seconds (30 min). If the needle is
outside `NEEDLE_NOT_ROTATED_ANGLE_MIN`–`NEEDLE_NOT_ROTATED_ANGLE_MAX`, the camera
is considered rotated and the extrinsic calibration is no longer trusted.

### Confirmation before alerting

A single rotated result is not reported, to avoid alerts from false detections:

1. The first rotated result schedules a re-check after `NEEDLE_RECHECK_DELAY`
   seconds (60 s).
2. Rotation is confirmed only after `NEEDLE_ROTATION_CONFIRM_COUNT` (3)
   consecutive rotated results, i.e. the first check plus two re-checks.
3. A re-check that finds the needle within range, or finds no needle, cancels the
   confirmation and nothing is sent.

Once confirmed:

- `rotated` is published on `camera_issue` every loop (not retained) until a
  check finds the needle within range again.
- The hash of `camera_extrinsics.json` is saved to
  `camera_extrinsics_bad_hash.txt` and `invalid` is published (retained) on
  `camera_calibration_ex`.

### Clearing the calibration status

The status returns to `valid` in either of these ways:

- **Recalibrate:** a new extrinsic calibration changes `camera_extrinsics.json`;
  the app detects the different hash, deletes the bad-hash file and publishes
  `valid`. Restart the service so the new extrinsics are used for measurements.
- **Clear command:** publish `clear` (**not retained**) to
  `machine/<DB_TABLE>/commands/clear_calibration`. The app deletes
  `camera_extrinsics_bad_hash.txt` through
  `scripts/clear_calibration_bad_hash.py`, resets its in-memory state and
  publishes `valid`. No restart is needed.

A retained `clear` command would be re-delivered on every reconnect, so always
send it without retain.

To delete the file by hand (takes effect after the next restart):

```bash
python3 scripts/clear_calibration_bad_hash.py
```

## MQTT Topics

All topics use `DB_TABLE` as the machine ID:

| Topic | Direction | Payload | Retained |
|---|---|---|---|
| `machine/<DB_TABLE>/status/heartbeat` | device → broker | `on` every `MQTT_HEARTBEAT_INTERVAL` s | no |
| `machine/<DB_TABLE>/commands/reset` | broker → device | `reset`; device replies `reset_success` | no |
| `machine/<DB_TABLE>/commands/clear_calibration` | broker → device | `clear` | no |
| `machine/<DB_TABLE>/status/esp32_issue` | device → broker | `issue` while the ESP32 is disconnected | no |
| `machine/<DB_TABLE>/status/camera_issue` | device → broker | `rotated` every loop while rotation is confirmed | no |
| `machine/<DB_TABLE>/status/camera_calibration_ex` | device → broker | `invalid` / `valid` | yes |

## Project Structure

```text
THREAD/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── measurement.py
├── calibration.py
├── database.py
├── serial_reader.py
├── mqtt_heartbeat.py
├── needle_angle_measure.py
├── hardware_utils.py
├── file_cleaner.py
├── best_Model.pt
├── single_needle_model.pt
├── camera_calibration.json
├── camera_extrinsics.json
├── auto_run.sh
├── auto_runner.sh
├── download_calibartion_app.sh
├── scripts/
│   ├── clear_calibration_bad_hash.py
│   └── create_sudoers_thread_modprobe.sh
├── Utils/
│   ├── auto_capture.py
│   ├── camera_UI.py
│   ├── check_model.py
│   ├── check_stitch_distance.py
│   ├── data_capturing.py
│   ├── data_capturing_2.py
│   ├── mqtt_reset_test.py
│   └── usb_camera.py
└── Testing/
    └── test1.py
```

## Main Modules

- `main.py`: Runtime orchestration, reset handling, database writes, image
  saving, camera reconnects, and measurement validation.
- `measurement.py`: YOLO inference, ROI filtering, mask processing,
  pixel-to-world measurement, MAD filtering, and rolling smoothing.
- `config.py`: Camera, model, measurement, ROI, serial, database, MQTT, offset,
  and cleanup settings.
- `serial_reader.py`: ESP32 serial reader and command sender.
- `database.py`: MySQL connection and measurement insertion.
- `mqtt_heartbeat.py`: MQTT heartbeat publisher and reset / clear calibration
  command listener.
- `needle_angle_measure.py`: Needle angle detection worker used to detect a
  rotated camera.
- `scripts/clear_calibration_bad_hash.py`: Deletes
  `camera_extrinsics_bad_hash.txt`; used by the clear calibration command and
  runnable by hand.
- `file_cleaner.py`: Deletes old saved annotation files.
- `hardware_utils.py`: Camera and ESP32 discovery helpers.
- `calibration.py`: ChArUco calibration helpers.

## Configuration Reference

Important values in `config.py`:

- `MODEL_PATH`: model used by the runtime, currently `single_needle_model.pt`.
- `STITCH_CLASS_ID`, `FABRIC_CLASS_ID`: YOLO class IDs used by the measurement
  code.
- `CONF_THRESH`, `IOU_THRESH`, `MAX_DETECTIONS`: YOLO inference settings.
- `FRAME_BUFFER`: rolling median buffer length for seam and width smoothing.
- `MIN_STITCHES`: minimum inlier count required before accepting a measurement.
- `MAX_PX_DISTANCE`: maximum image-space stitch-to-fabric-edge distance.
- `SKIP_CLUSTER`, `TWO_ROW_THRESHOLD_PX`: row selection behavior for seam
  allowance measurement.
- `OUTLIER_MIN_SAMPLES`, `OUTLIER_MAD_SCALE`: MAD filter controls.
- `ROI_ENABLED`, `ROI_X_MIN`, `ROI_X_MAX`, `ROI_Y_MIN`, `ROI_Y_MAX`: active
  measurement area.
- `Seam_lower_limit`, `Seam_upper_limit`: valid seam allowance range.
- `stitch_lower_limit`, `stitch_upper_limit`: valid stitch width range.
- `CONFIRM_CONSECUTIVE`, `CONFIRM_TOLERANCE_MM`: sustained high-value
  confirmation controls.
- `SEAM_LENGTH_OFFSET`, `STITCH_WIDTH_OFFSET`: post-measurement calibration
  offsets.
- `INFERENCE_INTERVAL`: seconds between processed inference frames.
- `SAVE_DIR`: root folder for annotated frame output.
- `FILE_RETENTION_HOURS`, `FILE_CLEANUP_INTERVAL_SECONDS`: saved file cleanup.
- `SHOW_WINDOWS`: enables/disables OpenCV display windows.
- `NEEDLE_ANGLE_CHECK_INTERVAL`: seconds between needle angle checks.
- `NEEDLE_NOT_ROTATED_ANGLE_MIN`, `NEEDLE_NOT_ROTATED_ANGLE_MAX`: needle angle
  range treated as not rotated (env overridable).
- `NEEDLE_ROTATION_CONFIRM_COUNT`, `NEEDLE_RECHECK_DELAY`: consecutive rotated
  checks required before alerting, and the delay between re-checks.

## Dependencies

Core dependencies are listed in `requirements.txt`:

- OpenCV with contrib modules
- Ultralytics YOLO
- NumPy, SciPy, Pillow, Matplotlib
- PyTorch, installed as required by Ultralytics
- paho-mqtt
- pyserial
- mysql-connector-python
- python-dotenv
- psutil
- PyYAML
- requests

## Output Data

Each valid stitch-count delta produces a database row with:

- `total_distance`: cumulative stitched distance in millimeters.
- `stitch_length`: measured stitch width in millimeters.
- `seam_allowance`: measured seam allowance in millimeters.

Annotated frames are saved under:

```text
saved_annotations/<YYYY-MM-DD_HH-MM-SS>/
```

Old saved files are removed by `FileCleanerThread` according to the retention
settings in `config.py`.

## License

This project is licensed under the MIT License.

## Contact

**Rish Wijewardhena**

- GitHub: [@RishWijewardhena](https://github.com/RishWijewardhena)
- Project Link: [https://github.com/RishWijewardhena/vision-textile-inspection](https://github.com/RishWijewardhena/vision-textile-inspection)
