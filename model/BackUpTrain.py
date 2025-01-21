import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from emotion_model import EmotionCNN
import numpy as np
from PIL import Image

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Define the dataset
class EmotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load label and pixel data
        label = int(self.data.iloc[idx, 0])
        pixels = np.array(self.data.iloc[idx, 1].split(), dtype=np.uint8).reshape(48, 48)

        # Convert to image for transformation
        image = Image.fromarray(pixels)

        if self.transform:
            image = self.transform(image)

        return image, label

# Data preparation
def get_dataloaders(train_csv, test_csv, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_dataset = EmotionDataset(train_csv, transform=transform)
    test_dataset = EmotionDataset(test_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Main script
if __name__ == "__main__":
    train_csv = 'train.csv'
    test_csv = 'test.csv'

    train_loader, test_loader = get_dataloaders(train_csv, test_csv, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

    # Save the trained model
    torch.save(model.state_dict(), 'emotion_model.pth')
