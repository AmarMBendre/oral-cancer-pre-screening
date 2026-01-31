import os
from pymongo import MongoClient
import datetime
import random

# CONFIG
MONGO_URI = "mongodb://localhost:27017/" 
DB_NAME = "oral_cancer_db"

def seed_data():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        
        print(f"Connecting to {DB_NAME}...")
        
        # 1. Create specific Collections explicitly
        users_col = db['users']
        screenings_col = db['screenings']
        
        # 2. Add Dummy Screening Data if empty
        if screenings_col.count_documents({}) == 0:
            print("Seeding dummy screenings...")
            
            # Ensure we have a patient
            patient = users_col.find_one({'role': 'patient'})
            if not patient:
                print("Error: No patient found. Please register a patient first via the app.")
                return

            dummy_screenings = []
            for i in range(3):
                dummy_screenings.append({
                    'patient_id': str(patient['_id']),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'pain_level': random.randint(1, 8),
                    'duration': '2 weeks',
                    'history': 'None',
                    'image_path': 'static/uploads/sample.jpg', # Placeholder
                    'prediction_label': 'Low Risk',
                    'confidence_score': 0.85,
                    'is_reviewed': False,
                    'doctor_comments': None
                })
            
            screenings_col.insert_many(dummy_screenings)
            print(f"Inserted {len(dummy_screenings)} dummy screenings into 'screenings' collection.")
        else:
            print("'screenings' collection already has data.")

        print("\nSUCCESS: Database populated. Refresh MongoDB Compass.")

    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    seed_data()
