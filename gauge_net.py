import os
import json
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
from gauge_dataset import GaugeDataset

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        # Excitation: Two FC layers with ReLU and sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        # Scale the input feature map
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, reduction=16):
        super(ResidualSEBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        return self.relu(out)

class GaugeNet(nn.Module):
    def __init__(self):
        super(GaugeNet2, self).__init__()
        
        # Example: Use residual blocks with increasing channels.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualSEBlock(64, 64),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            ResidualSEBlock(64, 128, downsample=True),
            ResidualSEBlock(128, 128),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            ResidualSEBlock(128, 256, downsample=True),
            ResidualSEBlock(256, 256),
            nn.MaxPool2d(2)
        )
        
        self.layer4 = nn.Sequential(
            ResidualSEBlock(256, 512, downsample=True),
            ResidualSEBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.regressor(x)
        return x

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # -----------------------------
    # 3. Define Data Transformations
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # Uncomment to add normalization:
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # -----------------------------
    # 4. Create Dataset and Split into Train/Test
    # -----------------------------
    dataset = GaugeDataset(image_dir=args.image_dir, json_file=args.json_file, transform=transform, box_only=args.box_only)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # -----------------------------
    # 5. Initialize Model, Loss, and Optimizer
    # -----------------------------
    model = GaugeNet2().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # -----------------------------
    # 6. Training Loop
    # -----------------------------
    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Training Loss: {avg_loss:.4f}")

        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Test Loss: {avg_test_loss:.4f}")

        # Adjust the learning rate based on test loss
        scheduler.step(avg_test_loss)

        model.eval()
        dummy_input = torch.randn(1, 3, 512, 512).to(device)
        scripted_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(scripted_model, args.save_model)
        print(f"Model saved as TorchScript to {args.save_model}")

    # Save the trained model
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a gauge reading CNN model.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Directory containing the gauge images (e.g., './images').")
    parser.add_argument('--json_file', type=str, default='rotations.json',
                        help="Path to the JSON file with rotation labels.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=20,
                        help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--save_model', type=str, default='gauge_net.pt',
                        help="Path to save the trained model.")
    parser.add_argument('--box_only', action='store_true',
                        help="if set, only train on bounding boxes")
    args = parser.parse_args()
    
    main(args)
