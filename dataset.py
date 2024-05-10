import os
import tarfile
import torch
import torchaudio
import numpy as np
from utils import AudioAugs
import random
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Audio processing parameters
sample_rate = 22050
duration = 3
samples_per_track = sample_rate * duration
n_mels = 128
n_fft = 2048
hop_length = 512


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class CustomAudioDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        
        if self.labels is not None:
            label = self.labels[idx]
            return feature, label
        return feature


def extract_tar_files(tar_path, extract_to):
    with tarfile.open(tar_path, 'r') as file:
        file.extractall(extract_to)
    print(f"Extracted contents to {extract_to}")

def load_and_convert_to_melspectrogram(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = torchaudio.transforms.Vol(waveform.size(0))(waveform)
    waveform = waveform[:, :samples_per_track] if waveform.size(1) > samples_per_track else torch.nn.functional.pad(waveform, (0, samples_per_track - waveform.size(1)))
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, n_mels=n_mels)(waveform)
    S_DB = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    return S_DB.squeeze(0).numpy()

def process_audio_files(audio_path, label_file_path, output_path):
    features = []
    # Load labels
    with open(label_file_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    # List all mp3 files, sort them to ensure order, and process
    filenames = [f for f in os.listdir(audio_path) if f.endswith(".mp3") and not f.startswith("._")]
    filenames.sort(key=lambda x: int(x.split('.')[0]))  # Sorting by numerical order assuming filename like '0.mp3', '1.mp3', etc.
    
    for filename in filenames:
        file_path = os.path.join(audio_path, filename)
        index = int(filename.split('.')[0])  # Get index from filename '0.mp3' -> 0
        if index < len(labels):  # Ensure we do not go out of bounds
            features.append(load_and_convert_to_melspectrogram(file_path))
        else:
            print(f"Skipping {filename} as it exceeds label file count.")
    
    # Ensure data and labels have the same length
    min_length = min(len(features), len(labels))
    features = features[:min_length]
    labels = labels[:min_length]

    np.savez(output_path, data=np.array(features), labels=np.array(labels))


def create_data_loader(data_path, batch_size=64):
    with np.load(data_path) as loaded_data:
        data = loaded_data['data']
        labels = loaded_data['labels']
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Adding a channel dimension
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def initialize_data_loader(dataset_dir, tar_paths, batch_size=64, split_ratio=0.8, augmentations=None):
    train_loader = None
    val_loader = None
    test_loader = None

    # Process training and validation data
    for tar_path, subdir in tar_paths:
        extract_tar_files(tar_path, dataset_dir)
        data_path = os.path.join(dataset_dir, subdir + '_features.npz')

        if not os.path.exists(data_path):
            audio_path = os.path.join(dataset_dir, subdir)
            label_file = os.path.join(dataset_dir, 'train_label.txt')  # General label file path
            process_audio_files(audio_path, label_file, data_path)

        if os.path.exists(data_path):
            with np.load(data_path) as data:
                features = data['data']
                labels = data['labels']




            # Split the dataset
            split_index = int(len(features) * split_ratio)
            train_features = features[:split_index]
            train_labels = labels[:split_index]
            val_features = features[split_index:]
            val_labels = labels[split_index:]

            # Convert to PyTorch tensors
            train_features_tensor = torch.tensor(train_features, dtype=torch.float32).unsqueeze(1)
            train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
            val_features_tensor = torch.tensor(val_features, dtype=torch.float32).unsqueeze(1)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

            # Creating TensorDatasets
            train_dataset = CustomAudioDataset(train_features_tensor, train_labels_tensor, transform=augmentations)
            val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

            # Creating DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Process test data
    test_tar_path = '/scratch/hy2611/ML_Competition/dataset/test_mp3s.tar'
    test_subdir = 'test_mp3s'
    extract_tar_files(test_tar_path, dataset_dir)
    test_audio_path = os.path.join(dataset_dir, test_subdir)
    test_data_path = os.path.join(dataset_dir, test_subdir + '_features.npz')

    if not os.path.exists(test_data_path):
        process_audio_files(test_audio_path, None, test_data_path)  # No labels for test data

    if os.path.exists(test_data_path):
        with np.load(test_data_path) as data:
            test_features = data['data']

        # Convert to PyTorch tensors
        test_features_tensor = torch.tensor(test_features, dtype=torch.float32).unsqueeze(1)

        # Creating TensorDataset and DataLoader for test data
        test_dataset = TensorDataset(test_features_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



