import matplotlib.pyplot as plt
import numpy as np

epochs = list(range(1, 14))
train_acc = [0.779, 0.962, 0.980, 0.980, 0.985, 0.985, 0.992,
             0.993, 0.989, 0.986, 0.992, 0.992, 0.992]
val_acc   = [0.991, 0.996, 0.998, 0.998, 0.998, 0.998, 0.996,
             0.996, 0.998, 0.998, 0.998, 0.996, 0.998]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(epochs, train_acc, 'o-', color='#2563eb', linewidth=2,
        markersize=6, label='Training Accuracy')
ax.plot(epochs, val_acc, 's--', color='#16a34a', linewidth=2,
        markersize=6, label='Validation Accuracy')

ax.fill_between(epochs, train_acc, alpha=0.1, color='#2563eb')
ax.fill_between(epochs, val_acc, alpha=0.1, color='#16a34a')

ax.axhline(y=0.998, color='#dc2626', linestyle=':', linewidth=1.5,
           label='Best Val Accuracy (99.82%)')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Audio Model Training and Validation Accuracy', fontsize=13,
             fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.7, 1.02)
ax.set_xticks(epochs)
ax.grid(True, alpha=0.3)

for i, (ta, va) in enumerate(zip(train_acc, val_acc)):
    if i == len(epochs)-1:
        ax.annotate(f'{va:.3f}', xy=(epochs[i], va),
                    xytext=(epochs[i]-0.5, va+0.005),
                    fontsize=8, color='#16a34a')

plt.tight_layout()
plt.savefig('figure_6_1_training_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure_6_1_training_accuracy.png")