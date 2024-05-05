import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
import wandb
import pandas as pd
from model import AudioResNet, BasicBlock
from torch.optim.lr_scheduler import StepLR
from dataset import initialize_data_loader
from test import *
from tqdm import tqdm
from utils import AudioAugs
from model_new import SoundNetRaw, Down

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
    
    os.environ["WANDB_API_KEY"] = "cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb"
    os.environ["WANDB_MODE"] = "online" #"dryrun"
    wandb.login(key='cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb')
    wandb.init(project="ML_Competition1",name='18_ADAM_step')
    #wandb.config.update(args)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup the path and batch size for data loaders
    dataset_dir = '/scratch/hy2611/ML_Competition/dataset'
    tar_paths = [('/scratch/hy2611/ML_Competition/dataset/train_mp3s.tar', 'train_mp3s')]
    batch_size = 64
    
    audio_augs = AudioAugs(['white_noise', 'random_gain', 'time_shift', ], fs=22050)

    # Initialize DataLoaders for training and validation using a split or separate data

    train_loader, val_loader, test_loader = initialize_data_loader(dataset_dir, tar_paths, batch_size=64, split_ratio=0.8, augmentations=audio_augs)

    # Define model, loss function, and optimizer
    model = AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes=4, num_mels=128, use_etf=True).to(device)
    #model = AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes=4, num_mels=128).to(device)
    #model = SoundNetRaw(nf=32, clip_length=3, embed_dim=128, n_layers=4, nhead=8, factors=[4, 4, 4, 4], n_classes=4, dim_feedforward=512).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=3e-2, weight_decay=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(),
    #                         lr=3e-4,
    #                         betas=[0.9, 0.99],
    #                         weight_decay=0)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                    max_lr=3e-4,
    #                                                    steps_per_epoch=len(train_loader),
    #                                                    epochs=100,
    #                                                    pct_start=0.1,
    #                                                    )
    model.to(device)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step() 

                # After training steps
        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})  # Merge dicts and index lr if needed

        # Evaluate accuracy on the validation set after each epoch
        val_accuracy = calculate_accuracy(model, val_loader, device)
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')
        wandb.log({"val_accuracy": val_accuracy})


    # Save model state dictionary
    torch.save(model.state_dict(), '/scratch/hy2611/ML_Competition/model_state_dict.pth')
    print("Model state dictionary saved.")

    # Generate predictions for the test set
    predictions = predict(model, device, test_loader)
    save_predictions_to_csv(predictions, '/scratch/hy2611/ML_Competition/test_predictions2.csv')
    print("Test predictions saved as CSV.")


if __name__ == "__main__":
    main()