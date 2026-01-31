"""
Initialize Operators Collection
--------------------------------
Creates the operators collection and seeds it with a default operator account.
Run this once to set up the system.
"""

from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "oral_screening_db"  # New database name

def init_operators():
    """Create operators collection with default operator"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        operators_collection = db['operators']
        
        # Check if default operator exists
        if operators_collection.find_one({'username': 'operator1'}):
            print("[INFO] Default operator already exists.")
            return
        
        # Create default operator
        default_operator = {
            'username': 'operator1',
            'password': 'clinic123',  # In production, use hashed passwords
            'full_name': 'Healthcare Operator',
            'role': 'operator',
            'facility_name': 'City Hospital OPD',
            'created_at': datetime.now(),
            'last_login': None
        }
        
        operators_collection.insert_one(default_operator)
        
        print("\n[SUCCESS] Operators collection initialized!")
        print(f"[INFO] Default operator created:")
        print(f"       Username: operator1")
        print(f"       Password: clinic123")
        print(f"       Facility: City Hospital OPD")
        print("\n[NOTE] Use these credentials to log into the system.\n")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    init_operators()
