from utils import AudioAugs
import os
import librosa
import pandas as pd
import numpy as np
import torch
import torchaudio

def load_audio_files_with_torchaudio(path, file_paths, augmentor):
    features = []
    for file_path in file_paths:
        full_path = os.path.join(path, file_path)
        waveform, sample_rate = torchaudio.load(full_path)
        waveform = waveform.mean(dim=0, keepdim=True)  # Ensure mono by averaging channels
        augmented_waveform, _ = augmentor(waveform.squeeze(0).numpy())
        augmented_waveform = torch.tensor(augmented_waveform, dtype=torch.float32).unsqueeze(0)
        mfccs = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(augmented_waveform)
        mfccs_mean = mfccs.mean(dim=2).squeeze(0).numpy()
        features.append(mfccs_mean)
    return features


def get_dataset(data_dir, apply_augmentation=True):
    """
    Load dataset and process it for classification task with optional augmentation.
    """
    train_audio_path = os.path.join(data_dir, 'train_mp3s')
    test_audio_path = os.path.join(data_dir, 'test_mp3s')
    label_file = os.path.join(data_dir, 'train_label.txt')
    
    labels = pd.read_csv(label_file, header=None, names=['file', 'label'])
    
    train_files = os.listdir(train_audio_path)
    test_files = os.listdir(test_audio_path)
    
    # Instantiate the augmentor
    augmentor = AudioAugs(k_augs=['flip', 'tshift', 'mulaw'], fs=22050) if apply_augmentation else None

    # Load and process audio files
    train_features = load_audio_files_with_augmentation(train_audio_path, train_files, augmentor) if apply_augmentation else load_audio_files(train_audio_path, train_files)
    test_features = load_audio_files(test_audio_path, test_files)  # Assume no augmentation for testing

    train_df = pd.DataFrame(train_features)
    train_df['label'] = labels['label'].values[:len(train_features)]  # Make sure labels align correctly

    test_df = pd.DataFrame(test_features)
    
    return train_df, test_df

# # Example usage
# data_dir = '/scratch/hy2611/ML_Competition/dataset'
# train_data, test_data = get_dataset(data_dir)
