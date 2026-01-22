"""
Database operations module for storing measurements
"""
import mysql.connector
from datetime import datetime
from config import DB_CONFIG, STATE_IDLE, STATE_RUNNING, LOG_DEBUG


class DatabaseHandler:
    """Handles MySQL database operations for stitch measurements"""
    
    def __init__(self, config=None):
        self.config = config or DB_CONFIG
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config['host'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            self.cursor = self.connection.cursor()
            if LOG_DEBUG:
                print(f"✅ Database connected to {self.config['host']}/{self.config['database']}")
            return True
        except mysql.connector.Error as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def ensure_table_exists(self):
        """Create table if it doesn't exist"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS `{self.config['table']}` (
            `time_stamp` DATETIME PRIMARY KEY,
            `total_distance` FLOAT,
            `stitch_length` FLOAT,
            `top_distance` FLOAT,
            `state` ENUM('IDLE', 'RUNNING') DEFAULT 'IDLE'
        )
        """
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            if LOG_DEBUG:
                print(f"✅ Table '{self.config['table']}' ready")
            return True
        except mysql.connector.Error as e:
            print(f"❌ Failed to create table: {e}")
            return False
    
    def insert_measurement(self, total_distance, stitch_length
                           , top_distance, state=STATE_RUNNING):
        """
        Insert a measurement record into the database
        
        Args:
            total_distance: Total fabric length in mm (stitch_count * stitch_length)
            stitch_length: Stitch width in mm (float)
            top_distance: Distance from top edge in mm (seam_length)
            state: Machine state ('IDLE' or 'RUNNING')
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        insert_query = f"""
        INSERT INTO `{self.config['table']}` 
        (`time_stamp`, `total_distance`, `stitch_length`, `top_distance`, `state`)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        try:
            self.cursor.execute(insert_query, (
                timestamp,
                float(total_distance),
                float(stitch_length),
                float(top_distance),
                state
            ))
            self.connection.commit()
            
            if LOG_DEBUG:
                print(f"📊 DB Insert: time={timestamp}, total={total_distance:.2f}mm, "
                      f"length={stitch_length}, seam={top_distance:.2f}mm, state={state}")
            return True
            
        except mysql.connector.Error as e:
            print(f"❌ Database insert failed: {e}")
            self.connection.rollback()
            return False
    
    # def insert_measurement_batch(self, measurements):
    #     """
    #     Insert multiple measurements at once
        
    #     Args:
    #         measurements: List of tuples (total_distance, stitch_length, top_distance, state)
    #     """
    #     if not self.connection or not self.connection.is_connected():
    #         if not self.connect():
    #             return False
        
    #     insert_query = f"""
    #     INSERT INTO `{self.config['table']}` 
    #     (`time_stamp`, `total_distance`, `stitch_length`, `top_distance`, `state`)
    #     VALUES (%s, %s, %s, %s, %s)
    #     """
        
    #     data = []
    #     for m in measurements:
    #         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #         data.append((
    #             timestamp,
    #             float(m[0]),
    #             int(m[1]),
    #             float(m[2]),
    #             m[3] if len(m) > 3 else STATE_RUNNING
    #         ))
        
    #     try:
    #         self.cursor.executemany(insert_query, data)
    #         self.connection.commit()
    #         if LOG_DEBUG:
    #             print(f"📊 Inserted {len(data)} records to database")
    #         return True
    #     except mysql.connector.Error as e:
    #         print(f"❌ Batch insert failed: {e}")
    #         self.connection.rollback()
    #         return False
    
    def get_latest_measurement(self):
        """Retrieve the most recent measurement"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        query = f"""
        SELECT `time_stamp`, `total_distance`, `stitch_length`, `top_distance`, `state`
        FROM `{self.config['table']}`
        ORDER BY `time_stamp` DESC
        LIMIT 1
        """
        
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return {
                    'timestamp': result[0],
                    'total_distance': result[1],
                    'stitch_length': result[2],
                    'top_distance': result[3],
                    'state': result[4]
                }
            return None
        except mysql.connector.Error as e:
            print(f"❌ Query failed: {e}")
            return None
        
    def delete_measurements(self,timestamp):
        """delete a specific meaurement by timestamp""" 
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
        
        delete_query = f"""
        DELETE FROM `{self.config['table']}`
        WHERE `time_stamp` = %s
        """
        
        try:
            self.cursor.execute(delete_query, (timestamp,))
            self.connection.commit()
            if LOG_DEBUG:
                print(f"🗑️ Deleted measurement at {timestamp}")
            return True
        except mysql.connector.Error as e:
            print(f"❌ Delete failed: {e}")
            self.connection.rollback()
            return False
           
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
        if LOG_DEBUG:
            print("🛑 Database connection closed")
    
    def __enter__(self):
        self.connect()
        self.ensure_table_exists()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test function
if __name__ == "__main__":
    print("Testing database connection...")
    
    with DatabaseHandler() as db:
        # Test insert
        success = db.insert_measurement(
            total_distance=250.5,
            stitch_length=10,
            top_distance=25.05,
            state=STATE_RUNNING
        )
        
        if success:
            print("✅ Insert successful")
            
            # Test retrieve
            latest = db.get_latest_measurement()
            if latest:
                print("✅ Latest measurement:", latest)

                #delete the test record
                db.delete_measurements(latest['timestamp'])
                print("🗑️Test record deleted")
            else:
                print("⚠️ No measurements found")
        else:
            print("❌ Insert failed")
    
    print("Test complete")