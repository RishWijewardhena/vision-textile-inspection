"""""
File cleanup module for removing old files based on retention policy.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from config import FILE_RETENTION_HOURS, FILE_CLEANUP_INTERVAL_SECONDS ,SAVE_DIR

class FileCleanerThread:


    def __init__(self,directory=SAVE_DIR, retention_hours=FILE_RETENTION_HOURS, check_interval=FILE_CLEANUP_INTERVAL_SECONDS):
        '''
        Initialize the FileCleanerThread.
        
        Args:
            directory self: Directory to monitor and clean up old files.
            retention_hours: Delete files older than thuis many hours. 
            check_interval: Interval in seconds between cleanup checks.
        '''

        self.directory = directory
        self.retention_hours = retention_hours
        self.check_interval = check_interval
        self.running = False
        self.thread=None


        print(f" File cleaner initialized: {directory}, Retention: {retention_hours} hours, Check Interval: {check_interval} seconds")
        print(f"   Directory: {directory}")
        print(f"   Retention: {retention_hours} hours")
        print(f"   Check interval: {check_interval} seconds")


    def _delete_old_files(self):
        '''
         """Delete files older than retention period"""
        
        self : object
        '''

        if not os.path.exists(self.directory):
            print(f"Directory {self.directory} does not exist. Skipping cleanup.")
            return
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=self.retention_hours)

        deleted_count = 0
        deleted_size = 0

        try:
            for filename in os.listdir(self.directory):
                filepath=os.path.join(self.directory,filename)

                if not os.path.isfile(filepath):
                    continue
                
                #get the file modification time
                file_mtime=datetime.fromtimestamp(os.path.getmtime(filepath))

                #delete if older than cutoff time
                if file_mtime<cutoff_time:
                    try:
                        file_size=os.path.getsize(filepath)
                        os.remove(filepath)
                        deleted_count+=1
                        deleted_size+=file_size
                        print(f"Deleted file: {filepath}, Size: {file_size} bytes")
                        print(f"🗑️ Deleted: {filename} (age: {current_time-file_mtime})")

                    except Exception as e:
                        print(f"Error deleting file {filepath}: {e}")

            if deleted_count>0:
                size_mb=deleted_size/(1024*1024)
                print(f"✅ Cleanup complete: {deleted_count} files deleted ({size_mb:.2f} MB freed)")

        except Exception as e:
            print(f"❌Error during cleanup: {e}")


    def _cleanup_loop(self):
        '''
        Internal method: Run the cleanup loop in a separate thread.
        '''

        while self.running:
            print("🧹 Running file cleanup...")
            self._delete_old_files()
           
           #sleep in small intervals to allow graceful shutdown
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)

        print("File cleaner thread stopped.")

    
    def start(self):
        '''
        start the cealnuo thread
        '''

        if self.running:
            print("File cleaner is already running.")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop, daemon=True) #deamon True to exit with main program 
        self.thread.start()
        print("File cleaner thread started.")
        return True
    
    def stop(self):
        '''
        Stop the cleanup thread
        '''

        if not self.running:
            print("File cleaner is not running.")
            return False
        
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5) #TIMEOUT to prevent hanging
            self.thread = None
        
        print("File cleaner stopped.")
        return True
    
    def force_cleanup(self):
        """Manually trigger cleanup immediately"""
        print("🧹 Manual cleanup triggered...")
        self._delete_old_files()

#STANDALONE TEST
if __name__=="__main__":
    cleaner=FileCleanerThread()
    cleaner.start()
    try:
        print("File cleaner is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping file cleaner...")
        cleaner.stop()