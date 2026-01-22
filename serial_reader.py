"""
Serial communication module for reading stitch count
"""
import serial
import threading
import time
from config import SERIAL_PORT, SERIAL_BAUDRATE, SERIAL_TIMEOUT, LOG_DEBUG


class SerialReader:
    """Reads stitch count from serial port in a separate thread"""
    
    def __init__(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.running = False
        self.thread = None
        self.latest_stitch_count = 0
        self.lock = threading.Lock()
        
    def connect(self):
        """Establish serial connection"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for connection to stabilize
            if LOG_DEBUG:
                print(f"✅ Serial connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"❌ Failed to connect to serial port {self.port}: {e}")
            return False
    
    def start_reading(self):
        """Start reading serial data in background thread"""
        if not self.serial_conn or not self.serial_conn.is_open:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        if LOG_DEBUG:
            print("🔄 Serial reading thread started")
        return True
    
    def _read_loop(self):
        """Background thread that continuously reads serial data"""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        try:
                            stitch_count = int(line)
                            with self.lock:
                                self.latest_stitch_count = stitch_count
                            if LOG_DEBUG:
                                print(f"📥 Serial received stitch count: {stitch_count}")
                        except ValueError:
                            if LOG_DEBUG:
                                print(f"⚠️ Invalid serial data (not integer): {line}")
                else:
                    time.sleep(0.01)  # Small delay to prevent busy-waiting
            except Exception as e:
                print(f"❌ Serial read error: {e}")
                time.sleep(0.1)
    
    def get_stitch_count(self):
        """Get the latest stitch count (thread-safe)"""
        with self.lock:
            return self.latest_stitch_count
    
    def reset_stitch_count(self):
        """Reset stitch count to zero"""
        with self.lock:
            self.latest_stitch_count = 0
        if LOG_DEBUG:
            print("🔄 Stitch count reset to 0")
    
    def stop(self):
        """Stop reading and close serial connection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        if LOG_DEBUG:
            print("🛑 Serial connection closed")
    
    def __enter__(self):
        self.start_reading()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Test function
if __name__ == "__main__":
    print("Testing serial reader...")
    print(f"Attempting to read from {SERIAL_PORT}")
    
    with SerialReader() as reader:
        print("Reading for 30 seconds. Send stitch counts via serial...")
        for i in range(30):
            count = reader.get_stitch_count()
            print(f"Current stitch count: {count}")
            time.sleep(1)
    
    print("Test complete")