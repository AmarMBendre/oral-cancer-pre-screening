import sqlite3
import os

DB_NAME = "oral_cancer.db"

def initialize_database():
    """
    Initializes the SQLite database with two main tables:
    1. users: Stores login information for Patients and Doctors.
    2. screenings: Stores the oral cancer screening data, linking it to a patient.
    """
    
    # Remove old DB to ensure clean state for the new schema (Optional, but good for setup)
    # if os.path.exists(DB_NAME):
    #     os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # TABLE 1: USERS
    # Purpose: Handles Authentication and Role Management
    # ---------------------------------------------------------
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,  -- 'patient' or 'doctor'
            full_name TEXT
        )
    ''')

    # ---------------------------------------------------------
    # TABLE 2: SCREENINGS
    # Purpose: Stores the core project data (Image + AI Result + Doctor Feedback)
    # ---------------------------------------------------------
    # Note on Foreign Key: patient_id links to users.id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            timestamp TEXT,
            
            -- Clinical Inputs
            symptoms_data TEXT,    -- JSON string or comma-separated list of symptoms
            pain_level INTEGER,    -- 1-10
            duration TEXT,         -- e.g., '2 weeks'
            history TEXT,          -- Patient history text
            
            -- AI Inputs/Outputs
            image_path TEXT,
            prediction_label TEXT, -- 'High Risk' or 'Low Risk'
            confidence_score REAL,
            
            -- Doctor/Tele-medicine Section
            is_reviewed INTEGER DEFAULT 0, -- 0 = Pending, 1 = Reviewed
            doctor_comments TEXT,
            
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    ''')

    # ---------------------------------------------------------
    # SEED DATA (For Academic Demo)
    # Create one default patient and one default doctor
    # ---------------------------------------------------------
    
    # Check if users exist before adding to avoid duplicates
    cursor.execute("SELECT count(*) FROM users")
    if cursor.fetchone()[0] == 0:
        print("Seeding default users...")
        users = [
            ('patient1', '1234', 'patient', 'Rahul Patient'),
            ('doctor1', 'admin', 'doctor', 'Dr. Smith')
        ]
        cursor.executemany("INSERT INTO users (username, password, role, full_name) VALUES (?, ?, ?, ?)", users)
    
    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' initialized successfully with tables 'users' and 'screenings'.")

def view_data():
    """Helper to view data for debugging"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("\n--- USERS ---")
    for row in cursor.execute("SELECT * FROM users"):
        print(row)
        
    print("\n--- SCREENINGS ---")
    for row in cursor.execute("SELECT * FROM screenings"):
        print(row)
        
    conn.close()

if __name__ == "__main__":
    initialize_database()
    # view_data()