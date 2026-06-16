import matplotlib.pyplot as plt
import numpy as np

categories = ['LOW\nStress', 'MEDIUM\nStress', 'HIGH\nStress']
accuracy   = [93.2, 88.5, 96.1]
colors     = ['#16a34a', '#d97706', '#dc2626']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(categories, accuracy, color=colors, edgecolor='white',
              linewidth=1.5, width=0.5)

for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{acc}%', ha='center', va='bottom', fontsize=12,
            fontweight='bold')

ax.set_xlabel('Stress Level Category', fontsize=12)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax.set_title('Stress Level Classification Accuracy per Category',
             fontsize=13, fontweight='bold')
ax.set_ylim(80, 100)
ax.grid(True, axis='y', alpha=0.3)

legend_patches = [
    plt.Rectangle((0,0),1,1, color='#16a34a', label='Low Stress'),
    plt.Rectangle((0,0),1,1, color='#d97706', label='Medium Stress'),
    plt.Rectangle((0,0),1,1, color='#dc2626', label='High Stress'),
]
ax.legend(handles=legend_patches, fontsize=10)

plt.tight_layout()
plt.savefig('figure_8_3_stress_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_8_3_stress_classification.png")