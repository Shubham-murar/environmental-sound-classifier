import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_PATH = 'data/UrbanSound8K/audio'
SPEC_PATH = 'spectrograms/'

def create_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=None)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_db = librosa.power_to_db(mels, ref=np.max)

    plt.figure(figsize=(2.24, 2.24))  # 224x224 px image
    librosa.display.specshow(mels_db, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_all():
    metadata = pd.read_csv(CSV_PATH)
    for i, row in metadata.iterrows():
        fold = row['fold']
        file_name = row['slice_file_name']
        label = row['class']
        path = os.path.join(AUDIO_PATH, f'fold{fold}', file_name)
        output_dir = os.path.join(SPEC_PATH, label)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{file_name}.png')
        if not os.path.exists(output_file):
            create_spectrogram(path, output_file)

if __name__ == '__main__':
    generate_all()
