"""
Initialize Registration Codes Collection
------------------------------------------
Seeds the database with test registration codes for development/demo.
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
import secrets
import string

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "oral_cancer_db"

def generate_code():
    """Generate a unique code"""
    chars = string.ascii_uppercase + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(12))
    return f"REG-DOC-{random_part}"

def init_codes():
    """Initialize with 5 demo codes"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        codes_collection = db['registration_codes']
        
        # Check if codes already exist
        existing_count = codes_collection.count_documents({})
        if existing_count > 0:
            print(f"[INFO] Collection already has {existing_count} codes.")
            print("       Skipping initialization. Delete existing codes if you want fresh data.")
            return
        
        # Seed 5 demo codes
        demo_codes = []
        for i in range(5):
            code = generate_code()
            demo_codes.append({
                'code': code,
                'issued_at': datetime.now(),
                'expires_at': None,  # No expiry for demo
                'is_used': False,
                'used_by': None,
                'used_at': None,
                'issued_by': 'System',
                'notes': f'Demo code #{i+1}'
            })
        
        codes_collection.insert_many(demo_codes)
        
        print("\n[SUCCESS] Initialized registration_codes collection!")
        print(f"[INFO] Generated {len(demo_codes)} demo codes:\n")
        for i, doc in enumerate(demo_codes, 1):
            print(f"  {i}. {doc['code']}")
        
        print("\n[NOTE] Use these codes for testing doctor registration.\n")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    init_codes()
