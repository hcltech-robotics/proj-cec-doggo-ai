import os
import json
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights
from torch.utils.data import DataLoader, Dataset

from custom_transform import Noise

# ----------------------------
# Custom Dataset for Gauges
# ----------------------------
class GaugeDetectDataset(Dataset):
    def __init__(self, root, transforms=None):
        """
        Args:
            root (str): Root directory containing the data.
                Expected to contain images (rgb_XXXX.png), 
                bounding box numpy files (bounding_box_2d_tight_XXXX.npy),
                and JSON label files (bounding_box_2d_tight_labels_XXXX.json).
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transforms = transforms
        # List all image files (assumed to have extension .png)
        self.imgs = sorted([f for f in os.listdir(root) if f.endswith(".png")])

    def __getitem__(self, idx):
        # Load image
        img_filename = self.imgs[idx]
        img_path = os.path.join(self.root, img_filename)
        img = Image.open(img_path).convert("RGB")

        # Extract index from the filename (e.g., "rgb_0000.png" -> "0000")
        index = os.path.splitext(img_filename)[0].split("_")[-1]

        # Load bounding box data from numpy file
        bbox_filename = f"bounding_box_2d_tight_{index}.npy"
        bbox_path = os.path.join(self.root, bbox_filename)
        boxes_data = np.load(bbox_path)

        # Load JSON file with class labels
        labels_filename = f"bounding_box_2d_tight_labels_{index}.json"
        labels_path = os.path.join(self.root, labels_filename)
        with open(labels_path, "r") as f:
            labels_dict = json.load(f)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        # Process each bounding box record
        for rec in boxes_data:
            semantic_id = int(rec["semanticId"])
            key = str(semantic_id)
            if key not in labels_dict:
                continue  # Skip if no corresponding label

            class_info = labels_dict[key]["class"]
            # Assign label based on class info
            if "gauge_needle" in class_info:
                label = 2  # gauge handle (needle)
            elif "gauge" in class_info:
                label = 1  # gauge face
            else:
                continue  # Skip boxes with unwanted classes

            # Extract coordinates and compute area
            x_min, y_min, x_max, y_max = float(rec["x_min"]), float(rec["y_min"]), float(rec["x_max"]), float(rec["y_max"])
            if x_max <= x_min or y_max <= y_min:
                print(f"Skipping degenerate box: {[x_min, y_min, x_max, y_max]} in image {img_filename}")
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
            areas.append((x_max - x_min) * (y_max - y_min))
            iscrowd.append(0)  # assuming no crowd instances



         # If no boxes were found, create empty tensors with the correct shape
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        areas_tensor = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.empty((0,), dtype=torch.float32)
        iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor
        }

        if self.transforms:
            img = self.transforms(img)

        #print(f"file {img_filename}, target: {target}")

        return img, target, img_filename

    def __len__(self):
        return len(self.imgs)

# ----------------------------
# Model Definition
# ----------------------------
def get_model(num_classes, model_path = None, finetune=False, device=torch.device("cpu")):
    """
    Returns a Faster R-CNN model pre-trained on COCO, modified to detect num_classes objects.
    Args:
        num_classes (int): number of classes (including background). 
                           For gauge and gauge handle, use num_classes = 3.
    """
    # Load a pre-trained model with weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # Get the number of input features for the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if finetune and model_path and os.path.exists(model_path):
        print(f"Loading fine-tuned weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    elif finetune:
        print("No model weights found for fine-tuning!")
        exit(1)

    return model




# ----------------------------
# Transforms Definition
# ----------------------------
def get_transform(train):
    """
    Returns a composition of transforms:
      - Resizes the image to 512x512,
      - Converts the PIL image to a tensor,
      - Normalizes the image using ImageNet means and stds.
    """
    noise = Noise()

    return transforms.Compose([
        #transforms.Resize((512, 512)),
        noise,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ----------------------------
# Training Script
# ----------------------------
def main(image_dir, model_path, epochs, batch_size, num_workers, finetune, finetune_weights):
    # Create the dataset and data loader
    dataset = GaugeDetectDataset(root=image_dir, transforms=get_transform(train=True))
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: tuple(zip(*batch)),
        num_workers=num_workers
    )

    # Device configuration
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    # Number of classes: 0=background, 1=gauge face, 2=gauge handle
    num_classes = 3
    model = get_model(num_classes, model_path=finetune_weights, finetune=finetune, device=device)
    model.to(device)

    # Set up the optimizer (using SGD with Nesterov momentum)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)

    # Add a learning rate scheduler to decay the LR every few epochs (e.g., every 3 epochs)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Optional: if using warmup, implement a warmup scheduler here

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch = 0
        for images, targets, img_filenames in data_loader:
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{len(data_loader)}")
            batch += 1
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: compute losses
            #print(f"images: {img_filenames}, targets: {targets}")
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        # Step the LR scheduler at the end of each epoch
        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} -- Loss: {epoch_loss:.4f}")

        model.eval()
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_path)

        state_ditct_path = model_path + '.state_dict.pth'
        torch.save(model.state_dict(), state_ditct_path)
        print(f"Epoch complete. TorchScript model saved as '{model_path}'. State dict saved as '{state_ditct_path}'.")
    # Save the model using TorchScript for inference
   


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train gauge detection model.")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Directory containing gauge images.")
    parser.add_argument('--model_path', type=str, default="gauge_detect.pt",
                        help="Path to save the TorchScript model.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of DataLoader worker threads.")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of epochs for training.")
    parser.add_argument('--finetune', action='store_true',
                        help="Load existing model weights for fine-tuning.")
    parser.add_argument('--finetune_weights', type=str,
                        help="Weights used for fine-tuning.")

    args = parser.parse_args()
    
    main(args.image_dir, args.model_path, args.epochs, args.batch_size, args.num_workers, args.finetune, args.finetune_weights)
