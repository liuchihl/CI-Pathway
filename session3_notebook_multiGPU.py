# Install required packages (uncomment if running on a new environment)
# !pip install torch torchvision accelerate wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from accelerate import Accelerator
import wandb

# Initialize the Accelerator for distributed training
accelerator = Accelerator()

# Initialize WandB (only on the main process to avoid duplicate logging)
if accelerator.is_main_process:
    wandb.init(project='cifar10-distributed-training', config={
        'architecture': 'SimpleCNN3',
        'dataset': 'CIFAR-10',
        'epochs': 10,
        'batch_size': 128,
        'learning_rate': 0.001
    })

# Define transforms with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Prepare data loaders for distributed training
trainloader, testloader = accelerator.prepare(trainloader, testloader)

# Visualize a sample image (only on main process)
if accelerator.is_main_process:
    sample, label = next(iter(trainset))
    plt.imshow(np.transpose(np.array(sample), (1, 2, 0)) * 0.5 + 0.5)  # Denormalize for display
    plt.axis("off")
    plt.title(f'Label: {label}')
    plt.show()
# 3-layer CNN with Batch Normalization
class SimpleCNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleCNN3()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare model and optimizer for distributed training
model, optimizer = accelerator.prepare(model, optimizer)
# Training parameters
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        # Gather loss and predictions for metrics
        loss = accelerator.gather_for_metrics(loss).mean().item()
        _, predicted = torch.max(outputs.data, 1)
        predicted = accelerator.gather_for_metrics(predicted)
        labels = accelerator.gather_for_metrics(labels)

        running_loss += loss * inputs.size(0)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Gather loss and predictions for metrics
            loss = accelerator.gather_for_metrics(loss).mean().item()
            _, predicted = torch.max(outputs.data, 1)
            predicted = accelerator.gather_for_metrics(predicted)
            labels = accelerator.gather_for_metrics(labels)

            val_loss += loss * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / total
    val_acc = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Log metrics to WandB (only on main process)
    if accelerator.is_main_process:
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_accuracy': epoch_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        print(f'Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Finish WandB run
if accelerator.is_main_process:
    wandb.finish()

# Plot training and validation loss/accuracy (only on main process)
if accelerator.is_main_process:
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()