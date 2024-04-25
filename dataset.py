from utils import AudioAugs
import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import librosa
import librosa.display



def load_audio_files_with_librosa(path, file_paths, augmentor=None, feature_dir='/scratch/hy2611/ML_Competition/dataset/features'):
    # Ensure the directory for features exists, if not, create it
    os.makedirs(feature_dir, exist_ok=True)
    
    features = []
    for file_path in file_paths:
        if file_path.startswith('.'):  # Skip hidden/system files
            continue

        # Define the feature file path
        feature_file_name = os.path.splitext(file_path)[0] + '.npy'  # Change extension to .npy
        feature_path = os.path.join(feature_dir, feature_file_name)

        # Check if the feature file already exists
        if os.path.exists(feature_path):
            # Load the existing features
            mfccs_mean = np.load(feature_path)
        else:
            # Load and process the audio file if the feature does not exist
            full_path = os.path.join(path, file_path)
            waveform, sample_rate = librosa.load(full_path, sr=None)  # Load with original sample rate
            waveform = torch.from_numpy(waveform).float()  # Convert waveform to a tensor immediately after loading

            if augmentor:
                waveform = augmentor(waveform)  # Apply augmentation if any

            # Compute MFCCs and their mean
            mfccs = librosa.feature.mfcc(y=waveform.numpy(), sr=sample_rate, n_mfcc=13)  # Convert back if necessary for librosa
            mfccs_mean = np.mean(mfccs, axis=1)

            # Save the newly computed features to a file
            np.save(feature_path, mfccs_mean)  # Save the feature array as a NumPy binary file

        features.append(mfccs_mean)

    return features

def get_dataset(data_dir, apply_augmentation=True):
    """
    Load dataset and process it for classification task with optional augmentation.
    """
    train_audio_path = os.path.join(data_dir, 'train_mp3s')
    test_audio_path = os.path.join(data_dir, 'test_mp3s')
    label_file = os.path.join(data_dir, 'train_label.txt')
    
    # Read the label file into a pandas DataFrame
    labels = pd.read_csv(label_file, header=None, names=['file', 'label'])
    
    # Get a list of audio file names from the directories
    train_files = os.listdir(train_audio_path)
    test_files = os.listdir(test_audio_path)
    
    # Instantiate the augmentor if augmentation is to be applied
    augmentor = AudioAugs(k_augs=['flip'], fs=22050) if apply_augmentation else None

    # Load and process the audio files
    train_features = load_audio_files_with_librosa(train_audio_path, train_files, augmentor)
    test_features = load_audio_files_with_librosa(test_audio_path, test_files, augmentor=None)  # Assume no augmentation for testing

    # Create a DataFrame from the processed features
    train_df = pd.DataFrame(train_features)
    # Make sure labels align correctly with features
    train_df['label'] = labels['label'].values[:len(train_features)]

    test_df = pd.DataFrame(test_features)
    
    return train_df, test_df

def load_data():
    pass
    #TODO

# Ensure to handle the main execution here if this script is run as a standalone program
if __name__ == "__main__":
    data_dir = "path_to_your_data_directory"  # Change this to your actual data directory path
    train_df, test_df = get_dataset(data_dir)
    print("Training and testing datasets are ready.")
