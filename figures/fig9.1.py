import matplotlib.pyplot as plt
import numpy as np

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
N = len(categories)

models_data = {
    'Facial':        [87.3, 86.1, 88.0, 87.0],
    'Audio':         [82.6, 81.4, 83.5, 82.4],
    'Physiological': [91.2, 90.8, 91.5, 91.1],
    'Fusion':        [94.7, 94.2, 95.1, 94.6],
}

colors = ['#2563eb', '#d97706', '#dc2626', '#16a34a']
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

for (model, values), color in zip(models_data.items(), colors):
    vals = values + values[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=color)
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(75, 100)
ax.set_yticks([80, 85, 90, 95, 100])
ax.set_yticklabels(['80%', '85%', '90%', '95%', '100%'], fontsize=8)
ax.set_title('Model Performance Comparison — Radar Chart',
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_9_1_radar_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_9_1_radar_chart.png")