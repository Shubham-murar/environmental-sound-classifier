# 🔊 Environmental Sound Classifier

A deep learning model to classify real-world environmental sounds (gunshots, dog barks, sirens, etc.) using spectrograms and a CNN.

## 💡 What It Does
- Converts audio files to Mel Spectrograms
- Trains a CNN to classify sounds from the UrbanSound8K dataset
- Accepts new audio input for real-time predictions

## 🧠 Technologies Used
- Python, PyTorch
- librosa, torchaudio
- UrbanSound8K Dataset

## 📁 Folder Structure
- `models/`: CNN architecture
- `utils/`: Spectrogram generation
- `train.py`: Training script
- `predict.py`: Audio classification

## 📦 Installation
```bash
pip install -r requirements.txt
