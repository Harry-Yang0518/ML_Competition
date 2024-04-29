import os
import tarfile
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Audio processing parameters
sample_rate = 22050
duration = 3
samples_per_track = sample_rate * duration
n_mels = 128
n_fft = 2048
hop_length = 512

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
    with open(label_file_path, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    for filename in os.listdir(audio_path):
        if filename.endswith(".mp3") and not filename.startswith("._"):
            file_path = os.path.join(audio_path, filename)
            features.append(load_and_convert_to_melspectrogram(file_path))
    np.savez(output_path, data=np.array(features), labels=np.array(labels))

def create_data_loader(data_path, batch_size=64):
    with np.load(data_path) as loaded_data:
        data = loaded_data['data']
        labels = loaded_data['labels']
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # Adding a channel dimension
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def initialize_data_loader(dataset_dir, tar_paths, batch_size=64):
    for tar_path, subdir in tar_paths:
        extract_tar_files(tar_path, dataset_dir)
        data_path = os.path.join(dataset_dir, subdir + '_features.npz')
        if not os.path.exists(data_path):
            audio_path = os.path.join(dataset_dir, subdir)
            label_file = os.path.join(dataset_dir, subdir + '_label.txt')
            process_audio_files(audio_path, label_file, data_path)
    return create_data_loader(data_path, batch_size)



# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from pydub import AudioSegment
# import librosa

# # Constants for audio processing
# SAMPLE_RATE = 22050  # Sample rate
# DURATION = 3  # Duration of audio clips in seconds
# SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
# N_FFT = 2048  # FFT window size
# HOP_LENGTH = 512  # Number of samples between successive frames
# N_MELS = 128  # Number of Mel bands to generate

# # Path settings
# AUDIO_PATH = '/scratch/hy2611/ML_Competition/dataset/train_mp3s'  # Path where MP3s are stored
# TEST_PATH = '/scratch/hy2611/ML_Competition/dataset/test_mp3s'  # Path where test MP3s are stored
# WAV_PATH = '/scratch/hy2611/ML_Competition/dataset/wav'  # Path where WAVs should be saved
# TEST_WAV_PATH = '/scratch/hy2611/ML_Competition/dataset/test_wav'  # Path where test WAVs should be saved
# FEATURES = '/scratch/hy2611/ML_Competition/dataset/features'  # Path where features should be saved
# LABEL_FILE = '/scratch/hy2611/ML_Competition/dataset/train_label.txt'  # File containing all audio labels

# def ensure_dir(directory):
#     """Ensure that a directory exists, and if not, create it."""
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# def convert_mp3_to_wav(mp3_file, output_dir):
#     """Convert an MP3 file to a WAV file with adjusted sample rate and single channel, avoiding duplication."""
#     ensure_dir(output_dir)
#     base_filename = os.path.splitext(os.path.basename(mp3_file))[0] + '.wav'
#     wav_file = os.path.join(output_dir, base_filename)
#     if not os.path.exists(wav_file):
#         audio = AudioSegment.from_mp3(mp3_file)
#         audio = audio.set_frame_rate(SAMPLE_RATE)
#         audio = audio.set_channels(1)
#         audio.export(wav_file, format='wav')
#     return wav_file

# def save_features(features, file_path):
#     np.save(file_path, features)

# def load_features(file_path):
#     return np.load(file_path)

# def load_and_convert_to_melspectrogram(file_path):
#     y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
#     y = librosa.util.fix_length(y, size=SAMPLES_PER_TRACK)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
#     S_DB = librosa.power_to_db(S, ref=np.max)
#     return S_DB

# def load_and_process(file_path, output_dir, feature_dir):
#     wav_path = convert_mp3_to_wav(file_path, output_dir)
#     feature_path = os.path.join(feature_dir, os.path.splitext(os.path.basename(file_path))[0] + '.npy')
#     if os.path.exists(feature_path):
#         return load_features(feature_path)
#     else:
#         S_DB = load_and_convert_to_melspectrogram(wav_path)
#         save_features(S_DB, feature_path)
#         return S_DB

# def get_dataset(data_dir, batch_size=32, val_split=0.2):
#     ensure_dir(FEATURES)
#     label_file = os.path.join(data_dir, 'train_label.txt')
#     audio_path = os.path.join(data_dir, 'train_mp3s')
#     test_path = os.path.join(data_dir, 'test_mp3s')

#     labels = np.loadtxt(label_file).astype(int)
#     train_data = [load_and_process(os.path.join(audio_path, f'{i}.mp3'), WAV_PATH, FEATURES) for i in range(len(labels))]
#     train_data = np.array(train_data)

#     val_size = int(len(train_data) * val_split)
#     train_features, val_features = train_data[val_size:], train_data[:val_size]
#     train_labels, val_labels = labels[val_size:], labels[:val_size]

#     train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels))
#     val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels))

#     test_files = sorted(os.listdir(test_path))
#     test_data = [load_and_process(os.path.join(test_path, file), TEST_WAV_PATH, FEATURES) for file in test_files]
#     test_data = np.array(test_data)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(TensorDataset(torch.tensor(test_data)), batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader

# if __name__ == "__main__":
#     data_dir = '/scratch/hy2611/ML_Competition/dataset'
#     train_loader, val_loader, test_loader = get_dataset(data_dir)
#     print("Data loaders are ready.")
