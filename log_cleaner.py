"""
One-time log cleanup helper for removing old files from the logs directory.
"""

import os
from datetime import datetime, timedelta


def clean_old_logs(directory="logs", retention_days=30):
    """Delete log files older than the retention window."""

    if not os.path.exists(directory):
        print(f"Log cleanup skipped: directory not found -> {directory}")
        return

    cutoff_time = datetime.now() - timedelta(days=retention_days)
    deleted_files = 0
    deleted_bytes = 0

    print(f"🧹 Running log cleanup in {directory} (older than {retention_days} days)...")

    try:
        for root, _, files in os.walk(directory, topdown=False):
            for filename in files:
                file_path = os.path.join(root, filename)

                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                except OSError as exc:
                    print(f"⚠️ Could not read mtime for {file_path}: {exc}")
                    continue

                if mtime < cutoff_time:
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_files += 1
                        deleted_bytes += file_size
                        print(f"Deleted old log file: {file_path}")
                    except OSError as exc:
                        print(f"⚠️ Failed deleting {file_path}: {exc}")

            if root != directory:
                try:
                    if not os.listdir(root):
                        os.rmdir(root)
                        print(f"Removed empty log folder: {root}")
                except OSError:
                    pass

        if deleted_files:
            freed_mb = deleted_bytes / (1024 * 1024)
            print(f"Log cleanup complete: {deleted_files} files removed, {freed_mb:.2f} MB freed")
        else:
            print("Log cleanup complete: no files older than retention window")
    except Exception as exc:
        print(f"⚠️ Log cleanup error: {exc}")
