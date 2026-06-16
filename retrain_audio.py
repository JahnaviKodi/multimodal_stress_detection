import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TESS_PATH  = r"data/processed/audio/TESS Toronto emotional speech set data"
MODEL_PATH = r"models/audio_model.h5"

print("Loading TESS dataset from .npy files...")

X = []
y = []

emotion_map = {
    'angry':              'angry',
    'disgust':            'disgust',
    'fear':               'fear',
    'happy':              'happy',
    'neutral':            'neutral',
    'pleasant_surprise':  'surprise',
    'pleasant_surprised': 'surprise',
    'ps':                 'surprise',
    'sad':                'sad',
}

for folder in os.listdir(TESS_PATH):
    folder_path = os.path.join(TESS_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    folder_lower = folder.lower()
    emotion = None
    for key, val in emotion_map.items():
        if key in folder_lower:
            emotion = val
            break

    if emotion is None:
        print(f"Skipping: {folder}")
        continue

    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    print(f"  {folder}: {len(files)} files -> {emotion}")

    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            features = np.load(fpath)  # shape (40, 67)

            if features.ndim == 2:
                # Take mean across time axis -> (40,)
                features = np.mean(features, axis=1)
            elif features.ndim == 1:
                if len(features) > 40:
                    features = features[:40]
                elif len(features) < 40:
                    features = np.pad(features, (0, 40 - len(features)))

            features = features[:40].astype('float32')

            if len(features) != 40:
                continue

            X.append(features)
            y.append(emotion)
        except Exception as e:
            print(f"    Error {fname}: {e}")

print(f"\nTotal samples: {len(X)}")
if len(X) == 0:
    print("ERROR: No samples loaded!")
    exit()

counts = {}
for label in y:
    counts[label] = counts.get(label, 0) + 1
print(f"Emotion counts: {counts}")

X = np.array(X, dtype='float32')
y = np.array(y)

print(f"\nTraining feature range: min={X.min():.2f} max={X.max():.2f} mean={X.mean():.2f}")
print(f"Sample first 5 values: {X[0][:5]}")

# Encode labels
le          = LabelEncoder()
y_encoded   = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"\nClasses: {le.classes_}")
print(f"Num classes: {num_classes}")

y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

# Normalize
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

print(f"Normalized range: min={X_norm.min():.2f} max={X_norm.max():.2f}")

# Save normalization params
np.save('models/audio_mean.npy',    X_mean)
np.save('models/audio_std.npy',     X_std)
np.save('models/audio_classes.npy', le.classes_)
print("Saved normalization params")
print(f"Mean first 5: {X_mean[:5]}")
print(f"Std first 5:  {X_std[:5]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_onehot,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# Build improved model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40,)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]

print("\nTraining...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Classes: {le.classes_}")
print(f"Model saved to: {MODEL_PATH}")