import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Data Preprocessing Pipelines', fontsize=14, fontweight='bold')

pipelines = [
    {
        'title': 'Facial Pipeline',
        'color': '#2563eb',
        'bg': '#dbeafe',
        'steps': [
            'Raw Image\n(FER-2013)',
            'Grayscale\nConversion',
            'Face Detection\n(OpenCV Haar)',
            'Crop & Resize\n(48×48 px)',
            'Pixel Normalise\n(÷255)',
            'Model Input\n(48×48×1)',
        ]
    },
    {
        'title': 'Audio Pipeline',
        'color': '#d97706',
        'bg': '#fef3c7',
        'steps': [
            'Raw Audio\n(TESS .npy)',
            'Load Array\n(40×67)',
            'Mean across\nTime Axis',
            'Feature Vector\n(40,)',
            'Normalise\n(mean/std)',
            'Model Input\n(40,)',
        ]
    },
    {
        'title': 'Physiological Pipeline',
        'color': '#dc2626',
        'bg': '#fee2e2',
        'steps': [
            'WESAD CSV\nSensor Data',
            'Extract\nStatistics',
            'Feature Vector\n(40,)',
            'Normalise\nFeatures',
            'Binary Label\n(Stress/No)',
            'Model Input\n(40,)',
        ]
    }
]

for ax, pipeline in zip(axes, pipelines):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(pipeline['steps']) * 1.2 + 0.5)
    ax.axis('off')
    ax.set_title(pipeline['title'], fontsize=11, fontweight='bold',
                 color=pipeline['color'])

    for i, step in enumerate(pipeline['steps']):
        y = (len(pipeline['steps']) - i - 1) * 1.2 + 0.4
        rect = FancyBboxPatch((0.3, y), 3.2, 0.9, boxstyle="round,pad=0.1",
                              facecolor=pipeline['bg'],
                              edgecolor=pipeline['color'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.9, y+0.45, step, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color='#1a1a2e')

        if i < len(pipeline['steps']) - 1:
            y_arrow = y - 0.15
            ax.annotate('', xy=(1.9, y_arrow),
                        xytext=(1.9, y_arrow + 0.05),
                        arrowprops=dict(arrowstyle='->', lw=1.5,
                                       color=pipeline['color']))

plt.tight_layout()
plt.savefig('figure_4_1_preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_4_1_preprocessing_pipeline.png")