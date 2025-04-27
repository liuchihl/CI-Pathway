import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
print(torch.__version__)


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

sample, label = next(iter(trainset))

plt.imshow(np.array(sample))
plt.axis("off")
plt.show()


class LinearNN(nn.Module):
    def __init__(self, input_size=32*32*3, num_classes=10):
        super(LinearNN, self).__init__()
        
        # Define the model parts using linear layers
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)        # Second hidden layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
model = LinearNN()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

BATCH_SIZE = 64
# shuffle is to randomize the order of the data to avoid overfitting
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)



### Set the Loss Function ###
criterion = torch.nn.CrossEntropyLoss() # this is good for multi-class classification


### Load the Model ###
import torchvision.models as models
model = models.resnet18(weights="IMAGENET1K_V1")    # pretrained model
### Set the Device ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

### Set the Optimizer ###
learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

### Store metrics ###
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

### Training Loop ###
epochs = 10

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients since gradients are accumulated by default
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)  # Forward pass: computes the predictions
        loss = criterion(outputs, labels) # Compute the loss based on the difference between predictions and true labels
        # Backward pass and optimize
        loss.backward() # Backward pass: computes the gradients of the loss w.r.t. the model parameters
        optimizer.step() # Updates the model's parameters using the computed gradients
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)   # Get the predicted class, maximum for the best prediction       
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Store metrics
    train_loss = running_loss / len(trainloader)
    train_accuracy = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    # Training and validation loop
    with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Update metrics
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
    val_loss = val_running_loss / len(testloader)
    val_accuracy = 100.0 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    # Print epoch metrics
    print(f"Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
### WRITE TRAINING LOOP ###
### STORE THE PER EPOCH TRAIN/VAL LOSS/ACC IN THE ## 
### STORE METRICS LIST ABOVE ###



# Plot training and validation loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot training and validation loss
ax1.plot(range(1, epochs + 1), train_losses, label='Train Loss')
ax1.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Training and Validation Loss')

# Plot training and validation accuracy
ax2.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
ax2.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.set_title('Training and Validation Accuracy')

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'cifar10_model_dropout.pth')
# save the figures
plt.savefig('train_val_loss_dropout.png')