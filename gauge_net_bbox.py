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
from custom_transform import CLAHEPreprocess, Noise, ResizeWithPaddingAndBBox

import torch
import torch.nn as nn


# -----------------------------
# Squeeze-and-Excitation (SE) Block
# -----------------------------
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


# -----------------------------
# Residual Block with SE
# -----------------------------
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
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


class AttentionFusion(nn.Module):
    def __init__(self, img_dim, bbox_dim, proj_dim):
        super().__init__()
        # Project bbox features from bbox_dim to proj_dim
        self.bbox_proj = nn.Linear(bbox_dim, proj_dim)
        # The attention layer now expects img_dim + proj_dim features
        self.attention = nn.Linear(img_dim + proj_dim, 1)

    def forward(self, img_features, bbox_features):
        # Project bbox_features to match image feature dimensions (or any desired dimension)
        projected_bbox = self.bbox_proj(bbox_features)
        # Concatenate image features and projected bbox features
        combined = torch.cat([img_features, projected_bbox], dim=1)
        attn_scores = self.attention(combined)
        attn_scores = torch.sigmoid(attn_scores)  # Scalar weight per sample
        # Blend features with learned importance
        fused_features = attn_scores * img_features + (1 - attn_scores) * projected_bbox
        return fused_features


# -----------------------------
# Deeper GaugeNet with Enhanced Regressor
# -----------------------------
class GaugeNet(nn.Module):
    def __init__(self):
        super(GaugeNet, self).__init__()
        # # Initial stem similar to ResNet's first layers
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # (B, 64, H/2, W/2)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)    # (B, 64, H/4, W/4)

        # Example: Use residual blocks with increasing channels.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualSEBlock(64, 64),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            ResidualSEBlock(64, 128, downsample=True), ResidualSEBlock(128, 128), nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            ResidualSEBlock(128, 256, downsample=True), ResidualSEBlock(256, 256), nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            ResidualSEBlock(256, 512, downsample=True),
            ResidualSEBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Global average pooling to get a (B, 512, 1, 1) feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.attention_fusion = AttentionFusion(img_dim=512, bbox_dim=4, proj_dim=512)

        # Enhanced Regressor: combine CNN features (512) with bbox (4) to get a gauge reading.
        self.regressor = nn.Sequential(
            nn.Flatten(),  # flatten to shape (B, 512)
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x, bbox):
        """
        x: input image tensor of shape (B, 3, H, W) -- expects the cropped gauge face (e.g., 512x512)
        bbox: needle bounding box tensor of shape (B, 4) with normalized coordinates.
        """
        # # Initial stem
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # CNN backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling and flattening
        x = self.avgpool(x)
        img_features = x.view(x.size(0), -1)  # Shape: (B, 512)
        fused_features = self.attention_fusion(img_features, bbox)
        # Concatenate features with the bounding box (B, 4) -> (B, 516)
        output = self.regressor(fused_features)
        return output


# -----------------------------
# Main Training Routine
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    noise = Noise()
    clahe_transform = CLAHEPreprocess()
    resize_with_padding_transform = ResizeWithPaddingAndBBox()
    # 3. Define Data Transformations
    transform = transforms.Compose([noise, clahe_transform, resize_with_padding_transform])

    # 4. Create Dataset and Split into Train/Test
    # Note: Ensure your GaugeDataset now returns (image, needle_bbox, target)
    dataset = GaugeDataset(
        image_dir=args.image_dir,
        json_file=args.json_file,
        transform=transform,
        box_only=args.box_only,
        x_size=512,
        y_size=512,
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # 5. Initialize Model, Loss, and Optimizer
    model = GaugeNet().to(device)
    if args.finetune:
        model.load_state_dict(
            torch.load(args.finetune_weights, map_location=device, weights_only=True)
        )
        print(f"Loaded weights from {args.finetune_weights} for fine-tuning.")

    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 6. Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        # Update loop to unpack (images, bbox, targets)
        for images, bbox, targets in train_loader:
            print(f"Batch {batch+1}/{len(train_loader)}")
            batch += 1
            images = images.to(device)
            bbox = bbox.to(device)  # bbox shape: (B, 4)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images, bbox)
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
            for images, bbox, targets in test_loader:
                images = images.to(device)
                bbox = bbox.to(device)
                targets = targets.to(device)
                outputs = model(images, bbox)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Test Loss: {avg_test_loss:.4f}")

        # Adjust the learning rate based on test loss
        scheduler.step(avg_test_loss)

        # Optionally, save a TorchScript version of the model each epoch.
        model.eval()
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, args.save_model)

        state_dict_path = args.save_model + '_state_dict.pth'
        torch.save(model.state_dict(), state_dict_path)
        print(
            f"Epoch complete. TorchScript model saved as '{args.save_model}'. State dict saved as '{state_dict_path}'."
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a gauge reading CNN + Transformer model.')
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help="Directory containing the gauge images (e.g., './images').",
    )
    parser.add_argument(
        '--json_file',
        type=str,
        default='rotations.json',
        help="Path to the JSON file with rotation labels.",
    )
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        '--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        '--save_model', type=str, default='gauge_net.pt', help="Path to save the trained model."
    )
    parser.add_argument(
        '--box_only', action='store_true', help="If set, only train on bounding boxes"
    )
    parser.add_argument(
        '--finetune', action='store_true', help="Load existing model weights for fine-tuning."
    )
    parser.add_argument('--finetune_weights', type=str, help="Weights used for fine-tuning.")
    args = parser.parse_args()

    main(args)
