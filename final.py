import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from collections import OrderedDict

import tarfile

import pandas as pd
import numpy as np
from model import AudioResNet
from tqdm import tqdm

import os



print(os.listdir("/scratch/hy2611/ML_Competition/dataset"))
with np.load('/scratch/hy2611/ML_Competition/dataset/train_mp3s_features.npz') as train_data:
    features = train_data['data']
    labels = train_data['labels']

split_index = int(len(features) * 0.8)
train_features = features[:split_index]
train_labels = labels[:split_index]
val_features = features[split_index:]
val_labels = labels[split_index:]

train_features_tensor = torch.tensor(train_features, dtype=torch.float32).unsqueeze(1)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32).unsqueeze(1)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
with np.load('/scratch/hy2611/ML_Competition/dataset/test_mp3s_features.npz') as test_data:
    test_features = test_data['data']

test_features_tensor = torch.tensor(test_features, dtype=torch.float32).unsqueeze(1)

test_dataset = TensorDataset(test_features_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# model = AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes=4, num_mels=128).to(device)
# model = AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes=4, num_mels=128).to(device)
model = AudioResNet(1, 4).to(device)
model
print([x.shape for x in model.res1.parameters()])

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
model.to(device)

def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


num_epochs = 100
val_accuracy = 0
epoch = 1
while val_accuracy < 0.99:
    model.train()
    scheduler.step()
    i = 1
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 1:
            print(i, end=' ')
        i += 1
    
    val_accuracy = calculate_accuracy(model, val_loader, device)
    print(f'\nEpoch {epoch}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')
    if val_accuracy > 0.925:
        torch.save(model.state_dict(), f'/kaggle/working/model_state_dict_epoch_{epoch}.pth')
        print(f"Eopch {epoch} model state dictionary saved.")
    epoch += 1


torch.save(model.state_dict(), '/kaggle/working/model_state_dict_attempt_3.pth')
print("Model state dictionary saved.")

model.eval()
predictions = []

with torch.no_grad():
    print(len(test_loader))
    for data in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
        samples = data[0].to(device)
        outputs = model(samples)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())


df = pd.DataFrame({'id': range(len(predictions)), 'category': predictions})
df.to_csv('/kaggle/working/submition new 3.csv', index=False)
print("Test predictions saved as CSV.")