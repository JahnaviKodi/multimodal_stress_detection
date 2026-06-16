import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.6, 'Multimodal Stress Detection System Architecture',
        ha='center', va='center', fontsize=13, fontweight='bold')

# Layer backgrounds
layers = [
    (0.3, 5.2, 13.4, 1.8, '#dbeafe', 'FRONTEND LAYER (React.js)'),
    (0.3, 2.8, 13.4, 1.8, '#dcfce7', 'BACKEND LAYER (Flask API)'),
    (0.3, 0.4, 13.4, 1.8, '#fef3c7', 'MODEL LAYER'),
]
for x, y, w, h, color, label in layers:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='gray', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 0.2, y + h - 0.25, label, fontsize=9,
            fontweight='bold', color='#374151')

# Frontend components
frontend_boxes = [
    (1.0, 5.5, 'Camera\nFeed'),
    (3.5, 5.5, 'Audio\nRecorder'),
    (6.0, 5.5, 'Results\nDisplay'),
    (8.5, 5.5, 'History\nChart'),
    (11.0, 5.5, 'Report\nDownload'),
]
for x, y, label in frontend_boxes:
    rect = FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.05",
                          facecolor='#2563eb', edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.9, y + 0.45, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

# Backend components
backend_boxes = [
    (1.0, 3.1, 'Face\nAnalysis'),
    (3.5, 3.1, 'Audio\nProcessing'),
    (6.0, 3.1, 'rPPG Heart\nRate'),
    (8.5, 3.1, 'Late\nFusion'),
    (11.0, 3.1, 'Session\nStorage'),
]
for x, y, label in backend_boxes:
    rect = FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.05",
                          facecolor='#16a34a', edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.9, y + 0.45, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

# Model components
model_boxes = [
    (1.0, 0.7, 'DeepFace\nFacial Model'),
    (4.2, 0.7, 'MFCC Audio\nModel'),
    (7.4, 0.7, 'Physiological\nModel'),
    (10.6, 0.7, 'Weighted\nFusion'),
]
colors = ['#7c3aed', '#d97706', '#dc2626', '#0891b2']
for (x, y, label), color in zip(model_boxes, colors):
    rect = FancyBboxPatch((x, y), 2.2, 0.9, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 1.1, y + 0.45, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

# Arrows between layers
arrow_props = dict(arrowstyle='->', color='#374151', lw=1.5)
ax.annotate('', xy=(7, 2.8), xytext=(7, 5.2),
            arrowprops=dict(arrowstyle='<->', color='#374151', lw=2))
ax.annotate('', xy=(7, 0.4+0.9), xytext=(7, 2.8),
            arrowprops=dict(arrowstyle='<->', color='#374151', lw=2))

ax.text(7.2, 4.0, 'REST API', fontsize=8, color='#374151', style='italic')
ax.text(7.2, 1.8, 'Model\nInference', fontsize=8, color='#374151', style='italic')

plt.tight_layout()
plt.savefig('figure_3_1_system_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_3_1_system_architecture.png")