import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from emotion_model import EmotionCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = EmotionDataset(train_csv, transform=train_transform)
    test_dataset = EmotionDataset(test_csv, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    train_losses = []
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

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return train_losses

# Plot training loss
def plot_training_loss(train_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

# Visualize clusters using t-SNE
def visualize_clusters(model, data_loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Clusters of Emotion Features')
    plt.show()

# Main script
if __name__ == "__main__":
    logging.info("Starting training process...")

    train_csv = './train.csv'
    test_csv = './test.csv'

    train_loader, test_loader = get_dataloaders(train_csv, test_csv, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN().to(device)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)  # Adjust weights as needed
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = train_model(model, train_loader, criterion, optimizer, EPOCHS, device)

    # Save the trained model
    torch.save(model.state_dict(), './emotion_model.pth')

    # Print completion message
    logging.info("Training complete. Model saved as './emotion_model.pth'.")

    # Show training log
    plot_training_loss(train_losses)

    # Visualize clusters
    visualize_clusters(model, test_loader, device)
