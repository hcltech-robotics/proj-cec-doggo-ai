import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# -----------------------------
# 1. Define the Dataset Class
# -----------------------------
class GaugeDataset(Dataset):
    """
    Dataset for loading gauge images and their corresponding rotation labels.
    Assumes images are named "rgb_0000.png", "rgb_0001.png", etc., and that the
    JSON file contains objects with a "frame" (0-indexed) and "rotation".
    """
    def __init__(self, image_dir, json_file, transform=None, box_only=False):
        self.image_dir = image_dir
        self.transform = transform
        self.box_only = box_only

        if os.path.isabs(json_file):
            # Load the JSON data
            with open(json_file, 'r') as f:
                self.data = json.load(f)
        else:
            # Load the JSON data relative to the image directory
            with open(os.path.join(image_dir, json_file), 'r') as f:
                self.data = json.load(f)
        
        # Ensure consistent ordering
        self.data.sort(key=lambda x: x['frame'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        frame = entry['frame']
        rotation = entry['rotation']
        
        # Construct image filename (assuming frame 0 maps to "rgb_0001.png")
        filename = os.path.join(self.image_dir, f"rgb_{frame:04d}.png")
        image = Image.open(filename).convert('RGB')
        
        if self.box_only:
             # Load bounding box data from numpy file
            bbox_filename = f"bounding_box_2d_tight_{frame:04d}.npy"
            bbox_path = os.path.join(self.image_dir, bbox_filename)
            boxes_data = np.load(bbox_path)

            # Load JSON file with class labels
            labels_filename = f"bounding_box_2d_tight_labels_{frame:04d}.json"
            labels_path = os.path.join(self.image_dir, labels_filename)
            with open(labels_path, "r") as f:
                labels_dict = json.load(f)

            cropped_image = None
            # Process each bounding box record
            for rec in boxes_data:
                semantic_id = int(rec["semanticId"])
                key = str(semantic_id)
                if key not in labels_dict:
                    continue  # Skip if no corresponding label

                class_info = labels_dict[key]["class"]
                if "gauge" != class_info:
                    continue
                
                # Extract coordinates and compute area
                x_min, y_min, x_max, y_max = float(rec["x_min"]), float(rec["y_min"]), float(rec["x_max"]), float(rec["y_max"])
                
                # Crop the image using the bounding box
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                break
            
            image = cropped_image
            #image.save("cropped_image.png")

        if self.transform:
            image = self.transform(image)
        print(filename)
        print(image)
        # Return image and its rotation label as a float tensor.
        return image, torch.tensor([rotation], dtype=torch.float32)