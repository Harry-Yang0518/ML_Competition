import torchaudio.transforms as T
import random
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import math
import librosa
from scipy.sparse import coo_matrix
from scipy.signal import firwin, butter, lfilter

class AudioAugs:
    def __init__(self, augs, fs=22050, ir_path=None):
        self.augs = augs
        self.fs = fs
        self.ir_path = ir_path
        self.transformations = {
            'white_noise': self.add_white_noise,
            'time_shift': self.time_shift,
            'random_gain': self.random_gain,
            'pitch_shift': self.pitch_shift,
            'add_reverb': self.add_reverb,
            'low_pass_filter': self.low_pass_filter,
        }
        if ir_path:
            self.impulse_response = torchaudio.load(ir_path)[0]
        else:
            self.impulse_response = None
    
    def __call__(self, sample):
        if not self.augs:
            return sample
        aug_choice = random.choice(self.augs)
        return self.transformations[aug_choice](sample)

    def add_white_noise(self, sample):
        noise = torch.randn_like(sample) * 0.005
        return sample + noise
    
    def time_shift(self, sample):
        shift_amount = random.randint(-1000, 1000)  # Adjust range according to your needs
        return torch.roll(sample, shifts=shift_amount, dims=-1)

    def random_gain(self, sample):
        gain = random.uniform(0.75, 1.25)
        return sample * gain

  

    def pitch_shift(self, sample):
        n_steps = random.uniform(-2, 2)  # Number of semitones
        sample = sample.numpy()
        sample_shifted = librosa.effects.pitch_shift(sample, sr=self.fs, n_steps=n_steps)
        return torch.from_numpy(sample_shifted).float()

    def add_reverb(self, sample):
        if self.impulse_response is None:
            return sample  # No impulse response loaded, return unmodified sample
        reverb_signal = torch.nn.functional.conv1d(sample[None, ...], self.impulse_response[None, ...], padding='same')
        return reverb_signal[0]

    def low_pass_filter(self, sample):
        cutoff_freq = random.uniform(300, 3000)  # Cutoff frequency in Hz
        b, a = butter(N=4, Wn=cutoff_freq / (self.fs / 2), btype='low')
        return torch.tensor(lfilter(b, a, sample.numpy()), dtype=torch.float32)

