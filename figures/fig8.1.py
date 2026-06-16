import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cm = np.array([[189, 14],
               [18, 203]])

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Stress', 'Predicted: No Stress'],
            yticklabels=['Actual: Stress', 'Actual: No Stress'],
            annot_kws={'size': 16, 'weight': 'bold'},
            linewidths=2, ax=ax)

ax.set_title('Confusion Matrix — Fusion Model', fontsize=13,
             fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('Actual Label', fontsize=11)

ax.text(0.5, 0.5, 'TP\n189', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white',
        transform=ax.transData)

plt.tight_layout()
plt.savefig('figure_8_1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_8_1_confusion_matrix.png")