import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

ax.text(7, 4.7, 'Data Flow Diagram — Stress Detection Pipeline',
        ha='center', fontsize=13, fontweight='bold')

steps = [
    (0.3,  2.0, 1.8, 1.2, '#dbeafe', '#2563eb', 'User\nRecord\nVideo+Audio'),
    (2.5,  2.0, 1.8, 1.2, '#ede9fe', '#7c3aed', 'Browser\nCaptures\nFrames+Audio'),
    (4.7,  2.0, 1.8, 1.2, '#dcfce7', '#16a34a', 'POST to\nFlask\nBackend'),
    (6.9,  2.0, 1.8, 1.2, '#fef3c7', '#d97706', 'Process\n3 Modalities\nIndependently'),
    (9.1,  2.0, 1.8, 1.2, '#fee2e2', '#dc2626', 'Late\nFusion\nWeighted Avg'),
    (11.3, 2.0, 1.8, 1.2, '#f0fdf4', '#15803d', 'Result to\nBrowser\n+ Save JSON'),
]

for x, y, w, h, bg, border, label in steps:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#1a1a2e')

for i in range(len(steps)-1):
    x1   = steps[i][0] + steps[i][2]
    x2   = steps[i+1][0]
    y_mid = steps[i][1] + steps[i][3]/2
    ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle='->', lw=2, color='#374151'))

sub = [
    (6.4, 0.3, 1.4, 0.7, '#7c3aed', 'Facial\nAnalysis'),
    (8.1, 0.3, 1.4, 0.7, '#d97706', 'Audio\nAnalysis'),
    (9.8, 0.3, 1.4, 0.7, '#dc2626', 'rPPG\nHR Est.'),
]

for x, y, w, h, color, label in sub:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, label, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold')
    ax.annotate('', xy=(x+w/2, y+h),
                xytext=(7.8, 2.0),
                arrowprops=dict(arrowstyle='->', lw=1, color='#94a3b8'))

plt.tight_layout()
plt.savefig('figure_3_3_data_flow.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_3_3_data_flow.png")