import matplotlib.pyplot as plt
import os

# Data
risk_categories = ['Low Risk\n(0-39)', 'Moderate Risk\n(40-69)', 'High Risk\n(70-100)']
percentages = [45, 35, 20]
colors = ['#10b981', '#fbbf24', '#ef4444']  # Green, Yellow, Red

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart
bars = ax.bar(risk_categories, percentages, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, pct in zip(bars, percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pct}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Formatting
ax.set_ylabel('Percentage of Screenings (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Risk Category', fontsize=12, fontweight='bold')
ax.set_title('Risk Distribution Analysis\n(Total Screenings: 100)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 55)  # Slight increase to make room for labels
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend box
legend_text = (
    'Sample Data: 100 Pre-Screenings\n'
    'Green: Low Risk (No immediate concern)\n'
    'Yellow: Moderate Risk (Clinical evaluation advised)\n'
    'Red: High Risk (Urgent medical attention required)'
)
ax.text(0.95, 0.95, legend_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save
output_path = os.path.abspath('risk_distribution_chart.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {output_path}")
