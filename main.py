import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
import wandb
import pandas as pd
from model import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from dataset import initialize_data_loader
from test import *
from tqdm import tqdm
from utils import AudioAugs
from model_new import SoundNetRaw, Down
import torch
import torch.nn as nn
import torch.nn.functional as F


def mixup_data(x, y, alpha=0, device='cuda'):
    '''Compute the mixup data. Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation during validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy

def calculate_class_weights(train_loader):
    class_counts = torch.zeros(4, dtype=torch.float)  # Adjust the size depending on the number of classes
    for _, labels in train_loader:
        class_counts += torch.bincount(labels, minlength=class_counts.size(0))
    return class_counts

def main():
    
    os.environ["WANDB_API_KEY"] = "cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb"
    os.environ["WANDB_MODE"] = "online" #"dryrun"
    wandb.login(key='cd3fbdd397ddb5a83b1235d177f4d81ce1200dbb')
    wandb.init(project="ML_Competition_final",name='300ep_4layer')
    #wandb.config.update(args)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup the path and batch size for data loaders
    dataset_dir = '/scratch/hy2611/ML_Competition/dataset'
    tar_paths = [('/scratch/hy2611/ML_Competition/dataset/train_mp3s.tar', 'train_mp3s')]
    batch_size = 64
    
    #audio_augs = AudioAugs(['white_noise', 'random_gain', 'time_shift', 'low_pass_filter'], fs=22050)
    audio_augs = AudioAugs([], fs=22050)
    # Initialize DataLoaders for training and validation using a split or separate data

    train_loader, val_loader, test_loader = initialize_data_loader(dataset_dir, tar_paths, batch_size=64, split_ratio=0.8, augmentations=audio_augs)

    # Define model, loss function, and optimizer
    model = AudioResNet(1,4)
    #model = AudioResNet(base_model='resnet50', num_classes=4, use_etf=False)
    #model = AudioResNet(base_model='resnet101', num_classes=4, use_etf=False)

    #model = AudioResNet(BasicBlock, [2, 2, 2, 2], num_classes=4, num_mels=128, use_etf=True).to(device)
    #model = AudioResNet(BasicBlock, [3, 4, 6, 3], num_classes=4, num_mels=128,use_etf=False).to(device)
    #model = SoundNetRaw(nf=32, clip_length=3, embed_dim=128, n_layers=4, nhead=8, factors=[4, 4, 4, 4], n_classes=4, dim_feedforward=512).to(device)

    #criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    class_counts = calculate_class_weights(train_loader) #tensor([ 824., 2484., 2730., 3470.])
    total_samples = class_counts.sum()
    #breakpoint()
    class_weights = total_samples / class_counts
    alpha_tensor = class_weights / class_weights.sum()
    #breakpoint()
    #criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0, reduction='mean', device=device)

    #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    #criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(),
    #                         lr=3e-4,
    #                         betas=[0.9, 0.99],
    #                         weight_decay=0)


    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.0001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=200, pct_start=0.1)
    model.to(device)

    # Training loop
    # Training loop
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0, device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Apply Mixup Criterion
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        # Log training and validation results
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "loss": loss.item(),
            "lr": current_lr,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Step scheduler with validation loss
        #scheduler.step(val_loss)
        scheduler.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy}')



    # Save model state dictionary
    torch.save(model.state_dict(), '/scratch/hy2611/ML_Competition/model_state_dict.pth')
    print("Model state dictionary saved.")

    # Generate predictions for the test set
    predictions = predict(model, device, test_loader)
    save_predictions_to_csv(predictions, '/scratch/hy2611/ML_Competition/300ep_4layer.csv')
    print("Test predictions saved as CSV.")


if __name__ == "__main__":
    main()