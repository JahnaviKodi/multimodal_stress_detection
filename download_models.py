import gdown
import os

os.makedirs('models', exist_ok=True)

files = {
    'models/audio_model.h5': 'YOUR_AUDIO_MODEL_GDRIVE_ID',
    'models/audio_mean.npy': 'YOUR_AUDIO_MEAN_GDRIVE_ID',
    'models/audio_std.npy': 'YOUR_AUDIO_STD_GDRIVE_ID',
    'models/audio_classes.npy': 'YOUR_AUDIO_CLASSES_GDRIVE_ID',
}

for path, file_id in files.items():
    if not os.path.exists(path):
        print(f'Downloading {path}...')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', path, quiet=False)
        print(f'Downloaded {path}')

print('All models ready')
