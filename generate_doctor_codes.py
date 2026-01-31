"""
One-Time Doctor Registration Code Generator
--------------------------------------------
This script generates unique registration codes for doctor accounts.
Each code can only be used once.

Usage:
    python generate_doctor_codes.py --count 5
    python generate_doctor_codes.py --count 10 --expires-days 30
"""

import argparse
import secrets
import string
from datetime import datetime, timedelta
from pymongo import MongoClient

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "oral_cancer_db"

def generate_code():
    """Generate a unique 16-character registration code"""
    # Format: REG-DOC-XXXXXXXXXXXX (16 chars total after prefix)
    chars = string.ascii_uppercase + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(12))
    return f"REG-DOC-{random_part}"

def create_registration_codes(count=1, expires_days=None, notes=""):
    """
    Generate and store registration codes in MongoDB
    
    Args:
        count: Number of codes to generate
        expires_days: Optional expiry in days (None = no expiry)
        notes: Optional notes for the codes
    """
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        codes_collection = db['registration_codes']
        
        generated_codes = []
        current_time = datetime.now()
        
        for i in range(count):
            code = generate_code()
            
            # Ensure uniqueness
            while codes_collection.find_one({'code': code}):
                code = generate_code()
            
            code_doc = {
                'code': code,
                'issued_at': current_time,
                'expires_at': current_time + timedelta(days=expires_days) if expires_days else None,
                'is_used': False,
                'used_by': None,
                'used_at': None,
                'issued_by': 'Admin',
                'notes': notes
            }
            
            codes_collection.insert_one(code_doc)
            generated_codes.append(code)
        
        return generated_codes
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Generate doctor registration codes')
    parser.add_argument('--count', type=int, default=1, help='Number of codes to generate')
    parser.add_argument('--expires-days', type=int, default=None, help='Expiry in days (optional)')
    parser.add_argument('--notes', type=str, default='', help='Notes for the codes')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DOCTOR REGISTRATION CODE GENERATOR")
    print("="*60)
    print(f"Generating {args.count} code(s)...")
    
    codes = create_registration_codes(
        count=args.count,
        expires_days=args.expires_days,
        notes=args.notes
    )
    
    if codes:
        print(f"\n[SUCCESS] Generated {len(codes)} code(s):\n")
        for i, code in enumerate(codes, 1):
            print(f"  {i}. {code}")
        
        if args.expires_days:
            expiry = datetime.now() + timedelta(days=args.expires_days)
            print(f"\n[EXPIRY] Codes expire on: {expiry.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"\n[EXPIRY] Codes have no expiry date")
        
        print(f"\n[INFO] Codes saved to MongoDB collection: registration_codes")
        print("="*60 + "\n")
    else:
        print("\n[ERROR] Failed to generate codes.\n")

if __name__ == "__main__":
    main()
