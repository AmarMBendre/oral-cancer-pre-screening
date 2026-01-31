"""
AI-Based Oral Cancer Pre-Screening System
==========================================
Healthcare Facility Operator Interface

This system is designed for use by healthcare operators (nurses, interns)
during OPD or screening camps to assist in oral cancer risk assessment.

IMPORTANT: This is a PRE-SCREENING tool, NOT a diagnostic system.
Final diagnosis MUST be made by a qualified doctor.
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from functools import wraps
import os

# ==================== FLASK CONFIGURATION ====================

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# Upload Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==================== MONGODB CONNECTION ====================

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "oral_screening_db"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    operators_collection = db['operators']
    screenings_collection = db['screenings']
    print(f"[OK] Connected to MongoDB: {DB_NAME}")
except Exception as e:
    print(f"[ERROR] MongoDB Connection Error: {e}")
    db = None

# ==================== AI MODEL LOADING ====================

# ==================== AI MODEL LOADING ====================

SIMULATION_MODE = False

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    import numpy as np
    
    # Initialize the same architecture as train_model.py
    def load_pytorch_model(model_path):
        model = models.mobilenet_v2(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Load weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device

    model, device = load_pytorch_model('oral_cancer_model_v2.pth')
    
    # Image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print(f"[OK] PyTorch AI Model loaded successfully on {device}")
except Exception as e:
    print(f"[WARNING] PyTorch AI Model not found or error loading - Running in SIMULATION mode")
    print(f"  Error: {e}")
    SIMULATION_MODE = True
    model = None
    device = None
    preprocess = None

# ==================== GEMINI API SETUP ====================

# ==================== GEMINI API SETUP ====================

GEMINI_ENABLED = False

try:
    from google import genai
    from google.genai import types
    import os

    # Read Gemini API key from environment variable
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("[INFO] GEMINI_API_KEY not found. Health education module disabled.", flush=True)
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_ENABLED = True
        print("[OK] Gemini API initialized successfully", flush=True)

except Exception as e:
    print(f"[WARNING] Gemini API initialization failed: {e}", flush=True)
    GEMINI_ENABLED = False

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_composite_risk(ai_score, patient_data):
    """
    Combines AI prediction with clinical risk factors
    Returns: (composite_score, risk_level)
    """
    # Base score from AI (0-100)
    score = ai_score * 100
    
    # Clinical Adjustments
    # 1. Tobacco Use
    tobacco = patient_data.get('tobacco_use', 'No')
    if tobacco == '> 10 years': score += 20
    elif tobacco == '5-10 years': score += 12
    elif tobacco == '< 5 years': score += 6
    
    # 2. Betel Nut
    betel = patient_data.get('betel_nut', 'No')
    if betel == '> 10 years': score += 15
    elif betel == '5-10 years': score += 10
    elif betel == '< 5 years': score += 5
    
    # 3. Pain Level
    pain = int(patient_data.get('pain_level', 0))
    if pain >= 8: score += 15
    elif pain >= 5: score += 8
    elif pain >= 3: score += 4
    
    # 4. Age
    age = int(patient_data.get('patient_age', 30))
    if age > 60: score += 12
    elif age > 45: score += 7
    
    # 5. Symptom Duration
    duration = patient_data.get('symptom_duration', '< 2 weeks')
    if duration == '> 3 months': score += 18
    elif duration == '1-3 months': score += 12
    elif duration == '2-4 weeks': score += 6
    
    # Clamp score
    score = min(100, score)
    
    # Map to risk level
    if score >= 75:
        risk = "High Risk"
    elif score >= 40:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"
        
    return round(score, 1), risk

def generate_health_education(risk_level, patient_data):
    """
    Generate AI-powered health education content using Gemini API.
    Academic Value: Provides contextual patient awareness without medical advice.
    
    Safety Constraints:
    - Explicitly forbids treatment/diagnosis
    - Validates output for dangerous keywords
    - Includes fallback for API failures
    """
    # Fallback message (used if API unavailable or unsafe output detected)
    fallback_message = (
        "Tobacco and betel nut are known risk factors for oral health issues. "
        "Regular medical checkups and early detection are important for better health outcomes. "
        "Please consult a qualified healthcare professional for proper evaluation and guidance."
    )
    
    if not GEMINI_ENABLED:
        print("[INFO] Gemini API disabled, using fallback message", flush=True)
        return fallback_message
    
    try:
        print(f"[DEBUG] Generating health education for Risk: {risk_level}", flush=True)
        print(f"[DEBUG] Patient data: Tobacco={patient_data.get('tobacco_use')}, Betel={patient_data.get('betel_nut')}, Pain={patient_data.get('pain_level')}", flush=True)
        
        # Construct safe, constrained prompt
        prompt = f"""You are a health education assistant for an oral cancer pre-screening system used in hospitals.

CONTEXT:
- A patient has completed an AI-based pre-screening (NOT a diagnosis)
- Risk Level: {risk_level}
- Tobacco Use: {patient_data.get('tobacco_use', 'Not specified')}
- Betel Nut Use: {patient_data.get('betel_nut', 'Not specified')}
- Pain Level: {patient_data.get('pain_level', 'Not specified')}/10
- Symptom Duration: {patient_data.get('symptom_duration', 'Not specified')}
- Age: {patient_data.get('patient_age', 'Not specified')} years

TASK:
Provide a brief, educational summary (4-5 sentences) covering ONLY:
1. What the risk factors (tobacco, betel nut) generally mean for oral health
2. Why {risk_level} screening results may occur with these habits
3. General symptoms to be aware of in oral cancer screening
4. Importance of consulting a doctor for clinical examination

STRICT RULES:
- DO NOT provide treatment plans or suggest medicines
- DO NOT confirm or deny a cancer diagnosis
- DO NOT give clinical instructions (e.g., "take this test")
- Use simple, non-alarming language suitable for general public
- End with: "Please consult a healthcare professional for proper evaluation."

Output format: Plain paragraph (no markdown, no bullet points, no bold text).
"""
        
        print("[DEBUG] Calling Gemini API...", flush=True)
        # Call Gemini API (new syntax)
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        if not response.text:
            print("[ERROR] Gemini API returned empty response (text is None). possibly blocked by safety settings.")
            return fallback_message

        ai_text = response.text.strip()
        
        print(f"[DEBUG] Gemini API returned {len(ai_text)} characters", flush=True)
        print(f"[DEBUG] First 100 chars: {ai_text[:100]}...", flush=True)
        
        # Validate output for dangerous keywords
        dangerous_keywords = [
            'prescribe', 'dosage', 'mg', 'ml', 'treatment plan',
            'you have cancer', 'diagnosed with', 'chemotherapy',
            'radiation', 'surgery required', 'biopsy shows',
            'take this medicine', 'drug', 'medication schedule'
        ]
        
        for keyword in dangerous_keywords:
            if keyword.lower() in ai_text.lower():
                print(f"[WARNING] AI Health Education output contained dangerous keyword: {keyword}")
                print("          Using fallback message instead.")
                return fallback_message
        
        # Additional validation: check minimum length
        if len(ai_text) < 100:
            print(f"[WARNING] AI Health Education output too short ({len(ai_text)} chars). Using fallback.")
            return fallback_message
        
        print("[SUCCESS] Gemini API generated valid health education content")
        return ai_text
        
    except Exception as e:
        print(f"[ERROR] Gemini API call failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        print("        Using fallback health education message.")
        return fallback_message

def predict_oral_cancer(image_path, patient_data=None):
    """
    Predict oral cancer risk from image AND clinical factors
    Returns: (risk_level, confidence, composite_score)
    """
    if SIMULATION_MODE or model is None:
        import random
        ai_prob = random.uniform(0.1, 0.9)
    else:
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            input_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.item()
            
            # Model mapping: 0=cancer, 1=non_cancer
            # Convert to cancer probability
            ai_prob = 1.0 - prediction
            
        except Exception as e:
            print(f"Prediction error: {e}")
            ai_prob = 0.5
    
    # Calculate composite clinical risk
    if patient_data:
        composite_score, risk_level = calculate_composite_risk(ai_prob, patient_data)
    else:
        # Fallback if no data
        composite_score = ai_prob * 100
        if composite_score > 70: risk_level = "High Risk"
        elif composite_score > 30: risk_level = "Moderate Risk"
        else: risk_level = "Low Risk"
        
    return risk_level, ai_prob, composite_score

def login_required(f):
    """Decorator to protect routes requiring authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'operator_id' not in session:
            flash("Please log in to access this page", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/')
def home():
    """Landing page - redirect based on authentication status"""
    if 'operator_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Operator login"""
    # Redirect if already logged in
    if 'operator_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        operator = operators_collection.find_one({'username': username})
        
        if operator and operator['password'] == password:
            # Set session
            session['operator_id'] = str(operator['_id'])
            session['operator_name'] = operator['full_name']
            session['facility'] = operator.get('facility_name', 'Healthcare Facility')
            
            # Update last login
            operators_collection.update_one(
                {'_id': operator['_id']},
                {'$set': {'last_login': datetime.now()}}
            )
            
            flash(f"Welcome, {operator['full_name']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Operator registration"""
    # Redirect if already logged in
    if 'operator_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        facility_name = request.form['facility_name']
        designation = request.form['designation']
        
        # Check if username exists
        if operators_collection.find_one({'username': username}):
            flash('Username already exists. Please choose another.', 'warning')
            return redirect(url_for('register'))
            
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
            
        # Create operator
        new_operator = {
            'full_name': full_name,
            'username': username,
            'password': password, # In production, hash this!
            'facility_name': facility_name,
            'designation': designation,
            'created_at': datetime.now(),
            'last_login': None
        }
        
        operators_collection.insert_one(new_operator)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Clear session and logout"""
    operator_name = session.get('operator_name', 'Operator')
    session.clear()
    flash(f"Goodbye, {operator_name}. You have been logged out.", "info")
    return redirect(url_for('login'))

# ==================== OPERATOR DASHBOARD ====================

@app.route('/dashboard')
@login_required
def dashboard():
    """Main operator dashboard with stats and recent screenings"""
    # Get today's stats filtered by operator
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    operator_id = session['operator_id']
    
    total_today = screenings_collection.count_documents({
        'screened_at': {'$gte': today_start},
        'screened_by': operator_id
    })
    
    high_risk_today = screenings_collection.count_documents({
        'screened_at': {'$gte': today_start},
        'ai_risk_level': 'High Risk',
        'screened_by': operator_id
    })
    
    # Get recent screenings for this operator only
    recent_screenings = list(
        screenings_collection.find({'screened_by': operator_id}).sort('screened_at', -1).limit(10)
    )
    
    return render_template('dashboard.html',
                         total_today=total_today,
                         high_risk_today=high_risk_today,
                         recent=recent_screenings)

# ==================== SCREENING WORKFLOW ====================

@app.route('/new-screening', methods=['GET', 'POST'])
@login_required
def new_screening():
    """Patient data entry and image upload form"""
    if request.method == 'POST':
        try:
            # Collect patient information
            patient_name = request.form['patient_name']
            patient_age = int(request.form['patient_age'])
            patient_mobile = request.form.get('patient_mobile', '')
            
            # Risk factors
            tobacco_use = request.form.get('tobacco_use', 'No')
            betel_nut = request.form.get('betel_nut', 'No')
            
            # Clinical data
            pain_level = int(request.form['pain_level'])
            symptom_duration = request.form['symptom_duration']
            lesion_location = request.form['lesion_location']
            
            # Handle image upload
            if 'lesion_image' not in request.files:
                flash('No image file uploaded', 'danger')
                return redirect(request.url)
            
            file = request.files['lesion_image']
            
            if file.filename == '':
                flash('No image selected', 'danger')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Save image with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Run Multi-Factor AI prediction
                patient_data = {
                    'patient_name': patient_name,
                    'patient_age': patient_age,
                    'tobacco_use': tobacco_use,
                    'betel_nut': betel_nut,
                    'pain_level': pain_level,
                    'symptom_duration': symptom_duration
                }
                
                risk_level, ai_prob, composite_score = predict_oral_cancer(filepath, patient_data)
                
                # Generate AI health education (NEW FEATURE)
                health_education = generate_health_education(risk_level, patient_data)
                
                # Store screening data
                screening_doc = {
                    'patient_name': patient_name,
                    'patient_age': patient_age,
                    'patient_mobile': patient_mobile,
                    'tobacco_use': tobacco_use,
                    'betel_nut': betel_nut,
                    'pain_level': pain_level,
                    'symptom_duration': symptom_duration,
                    'lesion_location': lesion_location,
                    'image_path': filepath,
                    'ai_risk_level': risk_level,
                    'ai_confidence': round(float(ai_prob) * 100, 1),
                    'composite_risk_score': composite_score,
                    'health_education': health_education,  # NEW: LLM-generated content
                    'model_version': 'v2.0 (PyTorch)' if not SIMULATION_MODE else 'simulation',
                    'screened_by': session['operator_id'],
                    'screened_at': datetime.now(),
                    'facility': session['facility'],
                    'referred_to_doctor': True if risk_level == 'High Risk' else False,
                    'notes': ''
                }
                
                result = screenings_collection.insert_one(screening_doc)
                
                flash('Screening completed successfully!', 'success')
                return redirect(url_for('view_result', id=str(result.inserted_id)))
            
            else:
                flash('Invalid file type. Please upload an image (PNG, JPG, JPEG)', 'danger')
                return redirect(request.url)
        
        except Exception as e:
            flash(f'Error processing screening: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('screening_form.html')

@app.route('/result/<id>')
@login_required
def view_result(id):
    """Display AI screening result with prominent disclaimers"""
    try:
        screening = screenings_collection.find_one({'_id': ObjectId(id)})
        
        if not screening:
            flash('Screening record not found', 'warning')
            return redirect(url_for('dashboard'))
        
        return render_template('result.html', screening=screening)
    
    except Exception as e:
        flash(f'Error loading result: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/download_report/<id>')
@login_required
def download_report(id):
    """
    Generate and download a PDF screening report.
    Academic Value: Provides tangible documentation for medical records and referrals.
    """
    try:
        from fpdf import FPDF
        from io import BytesIO
        
        screening = screenings_collection.find_one({'_id': ObjectId(id)})
        
        if not screening:
            flash('Screening record not found', 'warning')
            return redirect(url_for('dashboard'))
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'AI-Based Pre-Screening Report', ln=True, align='C')
        pdf.ln(5)
        
        # Facility and Date
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"Facility: {screening.get('facility', 'Healthcare Facility')}", ln=True)
        pdf.cell(0, 8, f"Screening Date: {screening['screened_at'].strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.cell(0, 8, f"Anonymized Patient ID: PAT-{str(screening['_id'])[-8:].upper()}", ln=True)
        pdf.ln(10)
        
        # Add Oral Lesion Image (NEW)
        if screening.get('image_path'):
            try:
                image_path = screening['image_path']
                # Convert Windows path to forward slashes for consistency
                image_path = image_path.replace('\\', '/')
                
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, 'Oral Cavity Image:', ln=True)
                pdf.ln(2)
                
                # Add image (centered, max width 120mm to fit on page)
                # Calculate centering position
                page_width = 210  # A4 width in mm
                image_width = 120
                x_position = (page_width - image_width) / 2
                
                pdf.image(image_path, x=x_position, w=image_width)
                pdf.ln(5)
                
            except Exception as e:
                print(f"[WARNING] Could not add image to PDF: {e}")
                pdf.set_font('Arial', 'I', 9)
                pdf.cell(0, 6, '(Image could not be embedded)', ln=True)
                pdf.ln(5)
        
        # Risk Assessment
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'AI Risk Assessment:', ln=True)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f"Risk Level: {screening['ai_risk_level']}", ln=True)
        pdf.cell(0, 8, f"AI Confidence: {screening['ai_confidence']}%", ln=True)
        pdf.cell(0, 8, f"Composite Risk Score: {screening.get('composite_risk_score', 'N/A')}/100", ln=True)
        pdf.ln(10)
        
        # Clinical Factors (Anonymized)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Clinical Factors:', ln=True)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 7, f"Age: {screening['patient_age']} years", ln=True)
        pdf.cell(0, 7, f"Tobacco Use: {screening.get('tobacco_use', 'Not specified')}", ln=True)
        pdf.cell(0, 7, f"Betel Nut: {screening.get('betel_nut', 'Not specified')}", ln=True)
        pdf.cell(0, 7, f"Lesion Location: {screening.get('lesion_location', 'Not specified')}", ln=True)
        pdf.ln(15)
        
        # AI Health Education Summary (NEW)
        if screening.get('health_education'):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(16, 185, 129)  # Green color
            pdf.cell(0, 10, 'AI Health Education & Awareness:', ln=True)
            pdf.set_text_color(0, 0, 0)  # Back to black
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, screening['health_education'])
            pdf.ln(5)
            
            # Educational Disclaimer
            pdf.set_font('Arial', 'B', 10)
            pdf.set_text_color(251, 191, 36)  # Warning yellow
            pdf.cell(0, 8, 'EDUCATIONAL DISCLAIMER:', ln=True)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('Arial', '', 9)
            pdf.multi_cell(0, 5,
                "This information is generated for educational and awareness purposes only. "
                "It does NOT constitute medical advice, diagnosis, or treatment. "
                "Please consult a qualified healthcare professional for clinical evaluation."
            )
            pdf.ln(10)
        
        # Medical Disclaimer (Most Important)
        pdf.set_font('Arial', 'B', 12)
        pdf.set_text_color(200, 0, 0)  # Red color for emphasis
        pdf.cell(0, 10, 'IMPORTANT MEDICAL DISCLAIMER:', ln=True)
        pdf.set_text_color(0, 0, 0)  # Back to black
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, 
            "This report is for PRE-SCREENING PURPOSES ONLY and is NOT a medical diagnosis. "
            "The AI system provides risk assessment to assist in clinical triage. "
            "A qualified medical professional MUST perform a clinical examination and, if necessary, "
            "a biopsy to confirm any diagnosis. Do not use this report as a substitute for professional "
            "medical advice, diagnosis, or treatment."
        )
        pdf.ln(10)
        
        # Footer
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 5, 'Academic Research Project - Not Approved as a Medical Device', ln=True, align='C')
        pdf.cell(0, 5, 'Generated by AI-Based Oral Cancer Pre-Screening System', ln=True, align='C')
        
        # Output PDF
        pdf_output = pdf.output(dest='S').encode('latin1')
        
        from flask import Response
        response = Response(pdf_output, mimetype='application/pdf')
        response.headers['Content-Disposition'] = f'attachment; filename=screening_report_{str(screening["_id"])[-8:]}.pdf'
        
        return response
    
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'danger')
        return redirect(url_for('view_result', id=id))

@app.route('/ethics')
def ethics():
    """
    Medical Ethics & Safety information page.
    Academic Value: Demonstrates regulatory awareness and responsible AI usage.
    """
    return render_template('ethics.html')

# ==================== REPORTS & HISTORY ====================

@app.route('/history')
@login_required
def screening_history():
    """View all past screenings with filters"""
    # Get filter parameters
    risk_filter = request.args.get('risk', 'all')
    search_query = request.args.get('search', '')
    
    # Build MongoDB query with operator isolation
    query = {'screened_by': session['operator_id']}
    
    if risk_filter != 'all':
        query['ai_risk_level'] = risk_filter
    
    if search_query:
        query['$or'] = [
            {'patient_name': {'$regex': search_query, '$options': 'i'}},
            {'patient_mobile': {'$regex': search_query, '$options': 'i'}}
        ]
    
    # Get screenings
    screenings = list(
        screenings_collection.find(query).sort('screened_at', -1)
    )
    
    return render_template('history.html', 
                         screenings=screenings,
                         risk_filter=risk_filter,
                         search_query=search_query)

@app.route('/statistics')
@login_required
def statistics():
    """Analytics dashboard with risk distribution"""
    # Overall statistics for this operator
    operator_id = session['operator_id']
    total_screenings = screenings_collection.count_documents({'screened_by': operator_id})
    
    high_risk_count = screenings_collection.count_documents({'ai_risk_level': 'High Risk', 'screened_by': operator_id})
    moderate_risk_count = screenings_collection.count_documents({'ai_risk_level': 'Moderate Risk', 'screened_by': operator_id})
    low_risk_count = screenings_collection.count_documents({'ai_risk_level': 'Low Risk', 'screened_by': operator_id})
    
    # Calculate percentages
    high_risk_pct = round((high_risk_count / total_screenings * 100), 1) if total_screenings > 0 else 0
    moderate_risk_pct = round((moderate_risk_count / total_screenings * 100), 1) if total_screenings > 0 else 0
    low_risk_pct = round((low_risk_count / total_screenings * 100), 1) if total_screenings > 0 else 0
    
    return render_template('statistics.html',
                         total_screenings=total_screenings,
                         high_risk_count=high_risk_count,
                         moderate_risk_count=moderate_risk_count,
                         low_risk_count=low_risk_count,
                         high_risk_pct=high_risk_pct,
                         moderate_risk_pct=moderate_risk_pct,
                         low_risk_pct=low_risk_pct)

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# ==================== RUN APPLICATION ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

