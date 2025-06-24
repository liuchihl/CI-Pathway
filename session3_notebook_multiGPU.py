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
testset = torchvision.datasets.CIFAR-10(root='./data', train=False, download=True, transform=transform)

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