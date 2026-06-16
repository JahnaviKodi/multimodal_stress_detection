import requests
import os

os.makedirs('models', exist_ok=True)

files = {
    'models/audio_classes.npy': 'https://huggingface.co/JahnaviKodi/multimodal-stress-detection/resolve/main/audio_classes.npy',
    'models/audio_mean.npy': 'https://huggingface.co/JahnaviKodi/multimodal-stress-detection/resolve/main/audio_mean.npy',
    'models/audio_model.h5': 'https://huggingface.co/JahnaviKodi/multimodal-stress-detection/resolve/main/audio_model.h5',
    'models/audio_std.npy': 'https://huggingface.co/JahnaviKodi/multimodal-stress-detection/resolve/main/audio_std.npy',
}

for path, url in files.items():
    if not os.path.exists(path):
        print(f'Downloading {path}...')
        r = requests.get(url, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'Done: {path}')

print('All models ready')
