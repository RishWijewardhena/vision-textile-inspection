"""
Delete the camera extrinsics bad-hash state file (camera_extrinsics_bad_hash.txt).

The file holds the hash of the extrinsics file that was in use when a needle rotation
was confirmed. While it exists the app reports the calibration as "invalid", also after
a restart.

Run manually from the repo directory:
    python3 scripts/clear_calibration_bad_hash.py

The main app also calls clear_calibration_bad_hash() when "clear" is sent to
machine/<DEVICE_ID>/commands/clear_calibration, and then publishes "valid".
"""
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_HASH_FILE = os.path.join(REPO_DIR, "camera_extrinsics_bad_hash.txt")


def clear_calibration_bad_hash(path=BAD_HASH_FILE):
    """Delete the bad-hash file. Returns True if the file is gone afterwards."""
    try:
        os.remove(path)
        print(f"🗑️ Deleted calibration bad-hash file: {path}")
    except FileNotFoundError:
        print(f"ℹ️ Calibration bad-hash file not found, nothing to delete: {path}")
    except OSError as exc:
        print(f"❌ Failed to delete calibration bad-hash file {path}: {exc}")
        return False
    return True


if __name__ == "__main__":
    sys.exit(0 if clear_calibration_bad_hash() else 1)
