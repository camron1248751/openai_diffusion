import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Configuration
# input_image_path = "path/to/input/image.jpg"
# generated_images_folder = "path/to/generated/images/folder"
# random_images_folder = "path/to/random/images/folder"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class
class DiffusionDataset(Dataset):
    def __init__(self, generated_images_folder, random_images_folder):
        self.generated_images = [
            os.path.join(generated_images_folder, img) for img in os.listdir(generated_images_folder)
        ]
        self.random_images = [
            os.path.join(random_images_folder, img) for img in os.listdir(random_images_folder)
        ]

    def __len__(self):
        return len(self.generated_images) + len(self.random_images)

    def __getitem__(self, idx):
        if idx < len(self.generated_images):
            img_path = self.generated_images[idx]
            label = 0
        else:
            img_path = self.random_images[idx - len(self.generated_images)]
            label = 1

        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = transform(img)
        return img, label

# CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load dataset
dataset = DiffusionDataset(generated_images_folder, random_images_folder)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Initialize the model, loss, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
        # Training
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / total:.4f}, Accuracy: {train_accuracy:.2f}%")

    # Evaluation
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct / total
        print(f"Test Loss: {test_loss / total:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "simple_cnn.pth")
