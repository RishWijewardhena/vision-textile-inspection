"""
Database operations module for storing measurements
"""
import mysql.connector
from datetime import datetime
from config import DB_CONFIG, LOG_DEBUG


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

    def get_last_record_date(self):
        """Get the date of the last inserted record"""
        try:
            query = f"SELECT timestamp FROM `{self.config['table']}` ORDER BY timestamp DESC LIMIT 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return result[0].date()  # Return only the date part
            return None
        except Exception as e:
            print(f"⚠️ Could not fetch last record date: {e}")
            return None
    
    # def ensure_table_exists(self):
    #     """Create table if it doesn't exist"""
    #     create_table_query = f"""
    #     CREATE TABLE IF NOT EXISTS `{self.config['table']}` (
    #         `id` BIGINT(20) AUTO_INCREMENT PRIMARY KEY,
    #         `time_stamp` DATETIME(3) NOT NULL,
    #         `stitch_length` DECIMAL(10,3),
    #         `seam_allowance` DECIMAL(10,3),
    #         `total_distance` DECIMAL(12,3)
    #     )
    #     """
    #     try:
    #         self.cursor.execute(create_table_query)
    #         self.connection.commit()
    #         if LOG_DEBUG:
    #             print(f"✅ Table '{self.config['table']}' ready")
    #         return True
    #     except mysql.connector.Error as e:
    #         print(f"❌ Failed to create table: {e}")
    #       return False
    
    def insert_measurement(self, total_distance, stitch_length
                           , seam_allowance):
        """
        Insert a measurement record into the database
        
        Args:
            total_distance: Total fabric length in mm (stitch_count * stitch_length)
            stitch_length: Stitch width in mm (float)
            seam_allowance: Distance from top edge in mm (seam_length)
      
        """
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] #millisecond precision
        
        insert_query = f"""
        INSERT INTO `{self.config['table']}` 
        (`timestamp`, `stitch_length`, `seam_allowance`, `total_distance`)
        VALUES (%s, %s, %s, %s)
        """
        
        try:
            self.cursor.execute(insert_query, (
                timestamp,
                float(stitch_length),
                float(seam_allowance),
                float(total_distance),

            ))
            self.connection.commit()
            
            if LOG_DEBUG:
                print(f"📊 DB Insert: time={timestamp}, total={total_distance:.1f}mm, "
                      f"length={stitch_length:.1f}, seam={seam_allowance:.1f}mm")
            return True
            
        except mysql.connector.Error as e:
            print(f"❌ Database insert failed: {e}")
            self.connection.rollback()
            return False
    
    
    def get_latest_measurement(self):
        """Retrieve the most recent measurement"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        query = f"""
        SELECT `id`, `timestamp`, `stitch_length`, `seam_allowance`, `total_distance`
        FROM `{self.config['table']}`
        ORDER BY `timestamp` DESC
        LIMIT 1
        """
        
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'timestamp': result[1],
                    'stitch_length': result[2],
                    'seam_allowance': result[3],
                    'total_distance': result[4],
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
        WHERE `timestamp` = %s
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
        # self.ensure_table_exists() tables are already created uncomment this if the tabels are not created
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
            stitch_length=5.0,
            seam_allowance=15.0

        )
        
        if success:
            print("✅ Insert successful")
            
            # Test retrieve
            latest = db.get_latest_measurement()
            if latest:
                print("✅ Latest measurement:", latest)

                #delete the test record
                db.delete_measurements(latest['timestamp'])
                print("🗑️ Test record deleted")
            else:
                print("⚠️ No measurements found")
        else:
            print("❌ Insert failed")
    
    print("Test complete")