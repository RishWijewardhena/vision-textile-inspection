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
        self._last_reconnect_attempt = 0
        self._reconnect_interval = 5  # seconds
        self._buffer = ""
        self._max_buffer_size = 8192
        
    def connect(self):
        """Establish serial connection"""
        try:
            self.serial_conn = serial.Serial(
                port=config.SERIAL_PORT,
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
        
    def _try_reconnect(self):
        """Attempt reconnect with rate limiting to avoid busy-loop retries."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_interval:
            return
        self._last_reconnect_attempt = now
        print("[INFO] Serial port not available, trying reconnect...")
        if self.serial_conn:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self.connect()

    def read_serial_data(self):
        """Read serial bytes, keep partial lines, and return one parsed stitch count if available."""
        if not self.serial_conn or not self.serial_conn.is_open:
            self._try_reconnect()
            return None

        if self.serial_conn.in_waiting:
            try:
                data = self.serial_conn.read(self.serial_conn.in_waiting).decode("utf-8", errors="ignore")
                self._buffer += data

                # Keep memory bounded if no newline arrives for a long time.
                if len(self._buffer) > self._max_buffer_size:
                    self._buffer = self._buffer[-self._max_buffer_size:]

                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        try:
                            stitch_count = int(line)
                            return stitch_count
                        except ValueError:
                            print(f"[WARN] Non-integer serial line ignored: {line}")
                            continue
            except Exception as e:
                print(f"Warning: Serial read/decode error: {e}")
                try:
                    self.serial_conn.close()
                except Exception:
                    pass
                self.serial_conn = None
                self._buffer = ""
                self._try_reconnect()
        return None
    
    def _read_loop(self):
        """Background thread that continuously reads serial data"""
        while self.running:
            try:
                stitch_count = self.read_serial_data()
                if stitch_count is not None:
                    with self.lock:
                        self.latest_stitch_count = stitch_count
                    if LOG_DEBUG:
                        print(f"📥 Serial received stitch count: {stitch_count}")
                else:
                    time.sleep(0.01)  # Small delay to prevent busy-waiting
            except Exception as e:
                print(f"❌ Serial read error: {e}")
                self._try_reconnect()
                time.sleep(0.1)
    
    def get_stitch_count(self):
        """Get the latest stitch count (thread-safe)"""
        with self.lock:
            return self.latest_stitch_count
    
    
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
        for i in range(100):
            count = reader.get_stitch_count()
            print(f"Current stitch count: {count}")
            time.sleep(1)
    
    print("Test complete")