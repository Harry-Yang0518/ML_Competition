from model import SoundNetRaw
from Trainer import train_model
from dataset import get_dataset
import torch
from tqdm import tqdm
import pandas as pd


NUM_CLASSES = 4
EPOCHS = 200
BATCH_SIZE = 32 
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, device, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
            images = data[0].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

def save_predictions_to_csv(predictions, file_name):
    df = pd.DataFrame({'id': range(len(predictions)), 'category': predictions})
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    model = SoundNetRaw(
    nf=32,                             # Number of filters in the initial convolution layer
    clip_length=66150 // 256,          # Total samples (66150 for 3s at 22050 Hz) divided by the product of the downsampling factors
    embed_dim=128,                     # Embedding dimension
    n_layers=4,                        # Number of layers
    nhead=8,                           # Number of attention heads
    factors=[4, 4, 4, 4],              # Downsampling factors for each layer
    n_classes=4,                      # Number of classes (adjust based on your specific task)
    dim_feedforward=512                # Dimensionality of the feedforward network within the transformer layers
    )
    model.to(device)
    data_dir = '/scratch/hy2611/ML_Competition/dataset'
    train_model(data_dir, model, device)

    torch.save(model, "Limbo.pth")
    
    # test_loader = load_data(BATCH_SIZE,)[2] 
    _, test_loader = get_dataset(data_dir)
    predictions = predict(model, device, test_loader)
    save_predictions_to_csv(predictions, 'predictions.csv')