import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_report_mockup():
    # Setup figure
    fig, ax = plt.subplots(figsize=(8.5, 11)) # Standard Letter size ratio
    ax.set_axis_off()
    
    # Background
    rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                            linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    
    # === HEADER ===
    plt.text(0.5, 0.95, "ORAL CANCER PRE-SCREENING REPORT", 
             ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    
    plt.axhline(y=0.92, xmin=0.05, xmax=0.95, color='black', linewidth=1)
    
    # === PATIENT INFO ===
    plt.text(0.1, 0.88, "Patient ID: P-2026-001", fontsize=11)
    plt.text(0.1, 0.85, "Date: 2026-01-21", fontsize=11)
    plt.text(0.6, 0.88, " facility: City Dental Hub", fontsize=11)
    plt.text(0.6, 0.85, "Operator: Dr. A. Sharma", fontsize=11)
    
    # === IMAGE PLACEHOLDER ===
    rect_img = patches.Rectangle((0.1, 0.55), 0.35, 0.25, 
                                linewidth=1, edgecolor='gray', facecolor='#f0f0f0')
    ax.add_patch(rect_img)
    plt.text(0.275, 0.675, "[Uploaded Lesion\nImage]", 
             ha='center', va='center', fontsize=10, color='gray')
    
    # === AI ASSESSMENT BOX ===
    rect_risk = patches.Rectangle((0.5, 0.55), 0.4, 0.25, 
                                 linewidth=1, edgecolor='red', facecolor='#fff5f5')
    ax.add_patch(rect_risk)
    
    plt.text(0.7, 0.75, "AI RISK ASSESSMENT", 
             ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.text(0.52, 0.70, "Risk Level:", fontsize=11, fontweight='bold')
    plt.text(0.75, 0.70, "HIGH RISK", fontsize=11, fontweight='bold', color='red')
    
    plt.text(0.52, 0.65, "Confidence:", fontsize=11)
    plt.text(0.75, 0.65, "87.5%", fontsize=11)
    
    plt.text(0.52, 0.60, "Composite Score:", fontsize=11)
    plt.text(0.75, 0.60, "82/100", fontsize=11)
    
    # === CLINICAL FACTORS ===
    plt.text(0.1, 0.50, "CLINICAL RISK PROFILE", fontsize=12, fontweight='bold', color='darkblue')
    plt.axhline(y=0.49, xmin=0.1, xmax=0.9, color='lightgray', linewidth=1)
    
    factors = [
        "Patient Age: 55 years",
        "Tobacco Use: > 10 years (Chronic)",
        "Betel Nut: Occasional",
        "Pain Level: 7/10",
        "Symptom Duration: > 3 months"
    ]
    
    y_pos = 0.45
    for factor in factors:
        plt.text(0.12, y_pos, "• " + factor, fontsize=10)
        y_pos -= 0.03
        
    # === HEALTH EDUCATION ===
    plt.text(0.1, 0.25, "AI-GENERATED HEALTH EDUCATION", fontsize=12, fontweight='bold', color='darkgreen')
    plt.axhline(y=0.24, xmin=0.1, xmax=0.9, color='lightgray', linewidth=1)
    
    edu_text = (
        "Based on the analysis, long-term tobacco use is a primary risk factor.\n"
        "The reported pain level (7/10) and symptom duration (> 3 months)\n"
        "require immediate clinical attention. Please consult a specialist\n"
        "for a biopsy and detailed examination."
    )
    plt.text(0.12, 0.18, edu_text, fontsize=10, style='italic', wrap=True)
    
    # === DISCLAIMER ===
    rect_disc = patches.Rectangle((0.05, 0.05), 0.9, 0.08, 
                                 linewidth=1, edgecolor='orange', facecolor='#fffbf0')
    ax.add_patch(rect_disc)
    plt.text(0.5, 0.09, "⚠️ MEDICAL DISCLAIMER", 
             ha='center', va='center', fontsize=10, fontweight='bold', color='#cc7a00')
    plt.text(0.5, 0.065, "This is a pre-screening tool only. It does NOT provide a medical diagnosis.\nConsult a qualified doctor for professional advice.", 
             ha='center', va='center', fontsize=8)

    # Save
    plt.savefig('system_output_sample.png', dpi=300, bbox_inches='tight')
    print("Report mockup created successfully.")

if __name__ == "__main__":
    create_report_mockup()
