import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
import pandas as pd
from model import AudioResNet, BasicBlock



def calculate_accuracy(model, data_loader, device):
    """Calculate the accuracy of a given model using a provided DataLoader."""
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare training data
    data_path = '/scratch/hy2611/ML_Competition/dataset/train_features.npz'
    with np.load(data_path) as data:
        features = data['data']
        labels = data['labels']

    # Splitting the dataset
    train_features = features[:8000]
    train_labels = labels[:8000]
    val_features = features[8000:]
    val_labels = labels[8000:]

    # Convert to PyTorch tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32).unsqueeze(1)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).unsqueeze(1)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Creating TensorDatasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)

    # Creating DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Define model, loss function, and optimizer
    model = AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes=4, num_mels=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2*10**(-5))
    model.to(device)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate accuracy on the validation set after each epoch
        val_accuracy = calculate_accuracy(model, val_loader, device)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')

    # Save model state dictionary
    torch.save(model.state_dict(), '/scratch/hy2611/ML_Competition/model_state_dict.pth')
    print("Model state dictionary saved.")

    # Predictions for submission (optionally)
    # Add prediction and submission code if needed

if __name__ == "__main__":
    main()
