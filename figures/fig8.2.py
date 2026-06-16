import matplotlib.pyplot as plt
import numpy as np

models    = ['Audio\nModel', 'Facial\nModel', 'Physiological\nModel', 'Fusion\nModel']
accuracy  = [82.6, 87.3, 91.2, 94.7]
precision = [81.4, 86.1, 90.8, 94.2]
recall    = [83.5, 88.0, 91.5, 95.1]
f1        = [82.4, 87.0, 91.1, 94.6]

x     = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 1.5*width, accuracy,  width, label='Accuracy',
               color='#2563eb', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision',
               color='#16a34a', alpha=0.85, edgecolor='white')
bars3 = ax.bar(x + 0.5*width, recall,    width, label='Recall',
               color='#d97706', alpha=0.85, edgecolor='white')
bars4 = ax.bar(x + 1.5*width, f1,        width, label='F1-Score',
               color='#dc2626', alpha=0.85, edgecolor='white')

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f'{h}%', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Performance Metrics: Single Modality vs Multimodal Fusion',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(70, 100)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)

# Highlight fusion bar
ax.axvline(x=2.5, color='#64748b', linestyle='--', alpha=0.5)
ax.text(3, 71, 'Fusion\nModel', ha='center', fontsize=9,
        color='#dc2626', fontweight='bold')

plt.tight_layout()
plt.savefig('figure_8_2_modality_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_8_2_modality_comparison.png")