"""
Backup data buffer for failed measurements
Persists to CSV when DB is unavailable, replays on reconnection
"""
import os
import csv
from collections import deque
from datetime import datetime


class BackupDataBuffer:
    """
    Manages failed database measurements with memory buffer + CSV backup
    - Keeps last N measurements in memory (fast access)
    - Persists to CSV when buffer fills (crash resilience)
    - Bulk replays to DB on reconnection
    - Removes successfully inserted measurements to prevent duplicates
    """
    
    def __init__(self, csv_path="backup_measurements.csv", max_memory=50):
        self.csv_path = csv_path
        self.max_memory = max_memory
        self.memory_buffer = deque(maxlen=max_memory)
        self.csv_header = ['timestamp', 'total_distance', 'stitch_length', 'seam_allowance']
        
        # Load any existing backup data from CSV
        self.load_from_csv()
    
    def add(self, timestamp, total_distance, stitch_length, seam_allowance):
        """
        Add a failed measurement to the backup buffer
        Persists to CSV if buffer reaches capacity
        """
        measurement = {
            'timestamp': timestamp,
            'total_distance': total_distance,
            'stitch_length': stitch_length,
            'seam_allowance': seam_allowance
        }
        
        self.memory_buffer.append(measurement)
        
        # Persist to CSV when buffer is full (overflow protection)
        if len(self.memory_buffer) >= self.max_memory:
            self._persist_to_csv()
            success=self._persist_to_csv()
            if success:
                self.memory_buffer.clear()   # Clear buffer after persisting to avoid duplicates on next startup
    
    def _persist_to_csv(self):
        """
        Write current buffer to CSV file
        Called when buffer reaches capacity (e.g., 50 items)
        """
        try:
            # Check if CSV already exists (don't duplicate)
            file_exists = os.path.exists(self.csv_path)
            
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write all measurements from buffer
                for measurement in self.memory_buffer:
                    writer.writerow(measurement)
            
            print(f"💾 Backed up {len(self.memory_buffer)} measurements to {self.csv_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to persist backup to CSV: {e}")
            return False
    
    def load_from_csv(self):
        """
        Load any previously failed measurements from CSV file
        Called on startup to recover from previous crashes
        """
        if not os.path.exists(self.csv_path):
            return
        
        try:
            with open(self.csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                count = 0
                for row in reader:
                    self.memory_buffer.append({
                        'timestamp': row['timestamp'],
                        'total_distance': float(row['total_distance']),
                        'stitch_length': float(row['stitch_length']),
                        'seam_allowance': float(row['seam_allowance'])
                    })
                    count += 1
            
            print(f"📂 Loaded {count} backed up measurements from {self.csv_path}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load backup CSV: {e}")
            return False
    
    def _update_buffer_with_failed(self, failed_measurements):
        """
        Update memory buffer and CSV with only failed measurements
        Removes successfully inserted items to prevent duplicates on next startup
        """
        # Clear current buffer and CSV
        self.memory_buffer.clear()
        
        # Delete old CSV file
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
            except Exception as e:
                print(f"⚠️ Failed to delete old backup file: {e}")
        
        # Repopulate buffer with only failed measurements
        for measurement in failed_measurements:
            self.memory_buffer.append(measurement)
        
        # Write only failed measurements to new CSV
        if failed_measurements:
            try:
                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
                    writer.writeheader()
                    for measurement in failed_measurements:
                        writer.writerow(measurement)
                
                print(f"📝 Updated backup file with {len(failed_measurements)} failed measurements")
                return True
            except Exception as e:
                print(f"❌ Failed to update backup CSV with failed measurements: {e}")
                return False
        
        return True
    
    def flush_to_db(self, db_handler):
        """
        Bulk insert all buffered measurements to database
        Tracks success/failure per measurement and keeps only failed items
        Returns True if all successful, False otherwise
        """
        if self.is_empty():
            return True
        
        all_measurements = list(self.memory_buffer)
        failed_measurements = []
        successful_count = 0
        
        try:
            for measurement in all_measurements:
                success = db_handler.insert_measurement(
                    total_distance=measurement['total_distance'],
                    stitch_length=measurement['stitch_length'],
                    seam_allowance=measurement['seam_allowance']
                )
                if not success:
                    failed_measurements.append(measurement)
                else:
                    successful_count += 1
            
            if len(failed_measurements) == 0:
                print(f"✅ Successfully flushed {len(all_measurements)} backed up measurements to DB")
                # Delete CSV and clear buffer only if ALL items were successful
                self._cleanup()
                return True
            else:
                # Partial flush: keep only failed measurements and update CSV/buffer
                print(f"⚠️ Flushed {successful_count}/{len(all_measurements)} measurements. "
                      f"{len(failed_measurements)} failed - retaining for next attempt")
                self._update_buffer_with_failed(failed_measurements)
                return False
        except Exception as e:
            print(f"❌ Flush to DB failed: {e}")
            return False
    
    def _cleanup(self):
        """
        Delete CSV file and clear memory buffer after successful flush
        """
        # Delete CSV file if it exists
        if os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
                print(f"🗑️ Deleted backup file {self.csv_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete backup file: {e}")
        
        # Clear memory buffer
        self.memory_buffer.clear()
    
    def get_all(self):
        """Return all buffered measurements as list"""
        return list(self.memory_buffer)
    
    def is_empty(self):
        """Check if buffer is empty"""
        return len(self.memory_buffer) == 0
    
    def is_full(self):
        """Check if buffer is at capacity"""
        return len(self.memory_buffer) >= self.max_memory
    
    def size(self):
        """Return current buffer size"""
        return len(self.memory_buffer)
