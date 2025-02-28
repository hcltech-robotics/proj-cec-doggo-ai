import os
import json
import argparse
import csv
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from gauge_dataset import GaugeDataset


# -----------------------------
# 2. Main Evaluation Function
# -----------------------------
def main(args):
    # Set the device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the image transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Adjust if necessary
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create the dataset and DataLoader.
    dataset = GaugeDataset(image_dir=args.image_dir, json_file=args.json_file, transform=transform, box_only=args.box_only)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load the saved model.
    # If your model is saved as a TorchScript module:
    model = torch.jit.load(args.model_path, map_location=device)
    # If your model is pickled, you might use:
    # model = torch.load(args.model_path, map_location=device)
    model.to(device)
    model.eval()

    # Evaluate and store predictions with corresponding labels.
    results = []
    
    with torch.no_grad():
        for images, bbox, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            bbox = bbox.to(device) 
            outputs = model(images, bbox)
            # Iterate over the batch and record each prediction and label.
            for pred, label in zip(outputs.cpu(), labels.cpu()):
                results.append((pred.item(), label.item()))
            print(len(results))

    # Save the results to CSV.
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Prediction', 'Label'])
        writer.writerows(results)
    print(f"Results saved to {args.output_csv}")


    error = 0
    for result in results:
        error += abs(result[0] - result[1])
        if abs(result[0] - result[1]) > 0.03:
            print(f"Error: {error}", "Result:", result)
    
    mse_error = None


    print(f"Mean error: {error/len(results)}")

# -----------------------------
# 3. Argparse for Command Line Arguments
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a saved gauge reading model on a folder of images.")
    parser.add_argument('--image_dir', type=str, required = True,
                        help="Directory containing gauge images (e.g., './images').")
    parser.add_argument('--json_file', type=str, default='rotations.json',
                        help="Path to the JSON file with rotation labels.")
    parser.add_argument('--model_path', type=str, default="gauge_net.pt",
                        help="Path to the saved model file (e.g., 'deep_gauge_net.pt').")
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help="Output CSV file to log predictions and labels.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for evaluation.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of DataLoader worker threads.")
    parser.add_argument('--box_only', action='store_true',
                        help="If set, then the bounding boxes will be evaluated only.")
    args = parser.parse_args()
    
    main(args)
