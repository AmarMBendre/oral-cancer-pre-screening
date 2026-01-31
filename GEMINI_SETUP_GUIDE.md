# AI Health Education Module - Quick Implementation Guide

## 🔑 Step 1: Get Gemini API Key (FREE)

1. Go to: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with "AIza...")

## 📝 Step 2: Set Environment Variable

### Windows (PowerShell):
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### Windows (Command Prompt):
```cmd
set GEMINI_API_KEY=YOUR_API_KEY_HERE
```

### Linux/Mac:
```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

## ✅ Step 3: Verification

Run the app:
```bash
python app.py
```

Look for this in console output:
```
[OK] Gemini API initialized successfully
```

If you see:
```
[INFO] Gemini API key not found. Health education module disabled.
```

Then the environment variable isn't set correctly.

## 🧪 Step 4: Testing

1. Login to the system
2. Create a new screening with:
   - Risk factors (tobacco, betel nut)
   - Pain level, symptom duration
3. View the result page
4. You should see a green **"🎓 AI Health Education & Awareness"** card
5. Download the PDF - it should include the education summary

## 🔄 Fallback Behavior

If the API fails or is unavailable, the system will show a safe fallback message instead:

> "Tobacco and betel nut are known risk factors for oral health issues. Regular medical checkups and early detection are important for better health outcomes. Please consult a qualified healthcare professional for proper evaluation and guidance."

## 💰 Cost Information

- **Free Tier**: 60 requests per minute
- **Typical Usage**: ~20 screenings per day
- **Monthly Cost**: $0 (under free tier limits)

## 🛡️ Safety Features Included

✅ Prompt engineering (no treatment advice)
✅ Keyword validation (blocks dangerous words)
✅ Fallback mechanism
✅ Visible disclaimers
✅ Educational framing only

## ❓ If You Get Errors

**Error: "API key not valid"**
→ Check that you copied the full key from Google AI Studio

**Error: "quota exceeded"**
→ Wait 1 minute (free tier resets every minute) or upgrade to paid

**No health education appears**
→ Check console logs for `[ERROR]` or `[WARNING]` messages
