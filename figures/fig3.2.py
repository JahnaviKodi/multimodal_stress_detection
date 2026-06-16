import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7, 5.7, 'Late Fusion Strategy — Multimodal Stress Detection',
        ha='center', fontsize=13, fontweight='bold')

# Input boxes
inputs = [
    (0.5, 3.5, '#dbeafe', '#2563eb', 'Video\nFrames'),
    (0.5, 2.0, '#dcfce7', '#16a34a', 'Audio\nRecording'),
    (0.5, 0.5, '#fef3c7', '#d97706', 'Heart Rate\n(rPPG)'),
]
for x, y, bg, border, label in inputs:
    rect = FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.1",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+0.9, y+0.45, label, ha='center', va='center',
            fontsize=9, fontweight='bold')

# Processing boxes
processing = [
    (3.0, 3.5, '#7c3aed', 'DeepFace\nEmotion Analysis\n(50%)'),
    (3.0, 2.0, '#d97706', 'MFCC Feature\nExtraction\n(30%)'),
    (3.0, 0.5, '#dc2626', 'HR-based\nStress Model\n(20%)'),
]
for x, y, color, label in processing:
    rect = FancyBboxPatch((x, y), 2.5, 0.9, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1.25, y+0.45, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')

# Score boxes
scores = [
    (6.5, 3.5, 'Facial\nStress Score'),
    (6.5, 2.0, 'Audio\nStress Score'),
    (6.5, 0.5, 'Physio\nStress Score'),
]
for x, y, label in scores:
    rect = FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.1",
                          facecolor='#f1f5f9', edgecolor='#64748b', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+0.9, y+0.45, label, ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1a1a2e')

# Fusion box
rect = FancyBboxPatch((9.2, 1.8), 2.0, 1.3, boxstyle="round,pad=0.1",
                      facecolor='#0891b2', edgecolor='white', linewidth=2)
ax.add_patch(rect)
ax.text(10.2, 2.45, 'Weighted\nAverage\nFusion', ha='center', va='center',
        fontsize=9, color='white', fontweight='bold')

# Output boxes
outputs = [
    (12.2, 3.5, '#dcfce7', '#16a34a', 'LOW'),
    (12.2, 2.2, '#fef3c7', '#d97706', 'MEDIUM'),
    (12.2, 0.9, '#fee2e2', '#dc2626', 'HIGH'),
]
for x, y, bg, border, label in outputs:
    rect = FancyBboxPatch((x, y), 1.5, 0.8, boxstyle="round,pad=0.1",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+0.75, y+0.4, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=border)

# Arrows
arrow_props = dict(arrowstyle='->', lw=1.5, color='#374151')
positions = [(3.5, 4.0), (3.5, 2.5), (3.5, 1.0)]
for (x, y), (ix, iy, _, _, _) in zip([(2.3, 3.95), (2.3, 2.45), (2.3, 0.95)], inputs):
    ax.annotate('', xy=(3.0, iy+0.45), xytext=(2.3, iy+0.45),
                arrowprops=arrow_props)

for x, y, label in scores:
    ax.annotate('', xy=(x, y+0.45), xytext=(x-0.7, y+0.45),
                arrowprops=arrow_props)
    ax.annotate('', xy=(9.2, 2.45), xytext=(x+1.8, y+0.45),
                arrowprops=dict(arrowstyle='->', lw=1, color='#94a3b8'))

ax.annotate('', xy=(12.2, 3.9), xytext=(11.2, 2.8),
            arrowprops=arrow_props)
ax.annotate('', xy=(12.2, 2.6), xytext=(11.2, 2.45),
            arrowprops=arrow_props)
ax.annotate('', xy=(12.2, 1.3), xytext=(11.2, 2.1),
            arrowprops=arrow_props)

plt.tight_layout()
plt.savefig('figure_3_2_fusion_strategy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_3_2_fusion_strategy.png")