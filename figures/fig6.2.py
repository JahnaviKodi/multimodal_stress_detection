import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Late Fusion Architecture', fontsize=14, fontweight='bold', pad=20)

# Input
inputs = [
    (0.5, 5.5, '#dbeafe', '#2563eb', 'Video Frames\n(224×224)'),
    (0.5, 3.3, '#fef3c7', '#d97706', 'Audio Recording\n(WebM)'),
    (0.5, 1.1, '#fee2e2', '#dc2626', 'Heart Rate\n(rPPG Est.)'),
]
for x, y, bg, border, label in inputs:
    rect = FancyBboxPatch((x, y), 2.2, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+1.1, y+0.5, label, ha='center', va='center',
            fontsize=9, fontweight='bold')

# Processing
processing = [
    (3.5, 5.5, '#7c3aed', 'DeepFace\nEmotion\nClassifier'),
    (3.5, 3.3, '#d97706', 'MFCC\nExtraction\n+ Dense NN'),
    (3.5, 1.1, '#dc2626', 'HR→Stress\nMapping\nFunction'),
]
for x, y, color, label in processing:
    rect = FancyBboxPatch((x, y), 2.2, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1.1, y+0.5, label, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')

# Stress scores
scores = [
    (6.5, 5.5, 'Facial Stress\nScore (50%)'),
    (6.5, 3.3, 'Audio Stress\nScore (30%)'),
    (6.5, 1.1, 'Physio Stress\nScore (20%)'),
]
for x, y, label in scores:
    rect = FancyBboxPatch((x, y), 2.2, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor='#f8fafc', edgecolor='#64748b', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+1.1, y+0.5, label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#1a1a2e')

# Fusion box
rect = FancyBboxPatch((9.5, 3.0), 2.2, 1.8,
                      boxstyle="round,pad=0.1",
                      facecolor='#0891b2', edgecolor='white', linewidth=2)
ax.add_patch(rect)
ax.text(10.6, 3.9, 'Weighted\nAverage\nFusion', ha='center', va='center',
        fontsize=10, color='white', fontweight='bold')

# Output
outputs = [
    (12.5, 5.5, '#dcfce7', '#16a34a', 'LOW\nStress'),
    (12.5, 3.3, '#fef3c7', '#d97706', 'MEDIUM\nStress'),
    (12.5, 1.1, '#fee2e2', '#dc2626', 'HIGH\nStress'),
]
for x, y, bg, border, label in outputs:
    rect = FancyBboxPatch((x, y), 1.3, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+0.65, y+0.5, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color=border)

# Draw arrows
arrow_props = dict(arrowstyle='->', lw=1.5, color='#374151')
for (ix, iy, _, _, _), (px, py, _, _) in zip(inputs, processing):
    ax.annotate('', xy=(px, py+0.5), xytext=(ix+2.2, iy+0.5),
                arrowprops=arrow_props)

for (px, py, _, _), (sx, sy, _) in zip(processing, scores):
    ax.annotate('', xy=(sx, sy+0.5), xytext=(px+2.2, py+0.5),
                arrowprops=arrow_props)

for sx, sy, _ in scores:
    ax.annotate('', xy=(9.5, 3.9), xytext=(sx+2.2, sy+0.5),
                arrowprops=dict(arrowstyle='->', lw=1.2,
                               color='#94a3b8',
                               connectionstyle='arc3,rad=0.1'))

for ox, oy, _, _, _ in outputs:
    ax.annotate('', xy=(ox, oy+0.5), xytext=(11.7, 3.9),
                arrowprops=dict(arrowstyle='->', lw=1.5,
                               color='#374151',
                               connectionstyle='arc3,rad=0.1'))

plt.tight_layout()
plt.savefig('figure_6_2_fusion_architecture.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_6_2_fusion_architecture.png")