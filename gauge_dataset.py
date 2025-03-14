import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class GaugeDataset(Dataset):
    """
    Dataset for loading gauge images and their corresponding rotation labels.
    Only includes data where both gauge and needle bounding boxes are present.
    """

    def __init__(
        self, image_dir, json_file, transform=None, box_only=False, x_size=None, y_size=None
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.box_only = box_only
        self.x_size = x_size
        self.y_size = y_size

        # Load the JSON data
        json_path = json_file if os.path.isabs(json_file) else os.path.join(image_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Ensure consistent ordering
        data.sort(key=lambda x: x['frame'])

        if box_only:
            # Pre-filter data to only include entries with valid bounding boxes.
            self.data = []
            for entry in data:
                frame = entry['frame']
                bbox_filename = f"bounding_box_2d_tight_{frame:04d}.npy"
                labels_filename = f"bounding_box_2d_tight_labels_{frame:04d}.json"
                bbox_path = os.path.join(self.image_dir, bbox_filename)
                labels_path = os.path.join(self.image_dir, labels_filename)
                if not (os.path.exists(bbox_path) and os.path.exists(labels_path)):
                    continue

                boxes_data = np.load(bbox_path, allow_pickle=True)
                with open(labels_path, "r") as f_labels:
                    labels_dict = json.load(f_labels)

                gauge_found = False
                needle_found = False
                for rec in boxes_data:
                    semantic_id = int(rec["semanticId"])
                    key = str(semantic_id)
                    if key not in labels_dict:
                        continue
                    class_info = labels_dict[key]["class"]
                    if class_info == "gauge":
                        gauge_found = True
                    elif class_info == "gauge,gauge_needle":
                        needle_found = True
                if gauge_found and needle_found:
                    self.data.append(entry)
        else:
            self.data = data
        print(f"Loaded {len(self.data)} entries from {image_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        frame = entry['frame']
        rotation = entry['rotation']

        # Construct image filename
        filename = os.path.join(self.image_dir, f"rgb_{frame:04d}.png")
        image = Image.open(filename).convert('RGB')

        # For box_only, crop image and compute normalized needle bbox.
        if self.box_only:
            bbox_filename = f"bounding_box_2d_tight_{frame:04d}.npy"
            labels_filename = f"bounding_box_2d_tight_labels_{frame:04d}.json"
            bbox_path = os.path.join(self.image_dir, bbox_filename)
            labels_path = os.path.join(self.image_dir, labels_filename)

            boxes_data = np.load(bbox_path, allow_pickle=True)
            with open(labels_path, "r") as f:
                labels_dict = json.load(f)

            cropped_image = None
            gauge_x_min = gauge_y_min = gauge_x_max = gauge_y_max = None
            needle_x_min = needle_y_min = needle_x_max = needle_y_max = None

            for rec in boxes_data:
                semantic_id = int(rec["semanticId"])
                key = str(semantic_id)
                if key not in labels_dict:
                    continue
                class_info = labels_dict[key]["class"]
                if class_info == "gauge":
                    gauge_x_min = int(rec["x_min"])
                    gauge_y_min = int(rec["y_min"])
                    gauge_x_max = int(rec["x_max"])
                    gauge_y_max = int(rec["y_max"])
                    cropped_image = image.crop(
                        (gauge_x_min, gauge_y_min, gauge_x_max, gauge_y_max)
                    )
                elif class_info == "gauge,gauge_needle":
                    needle_x_min = int(rec["x_min"])
                    needle_y_min = int(rec["y_min"])
                    needle_x_max = int(rec["x_max"])
                    needle_y_max = int(rec["y_max"])

            # Since we pre-filtered, these values should be set.
            gauge_width = gauge_x_max - gauge_x_min
            gauge_height = gauge_y_max - gauge_y_min
            needle_bbox = [
                (needle_x_min - gauge_x_min) / gauge_width,
                (needle_y_min - gauge_y_min) / gauge_height,
                (needle_x_max - gauge_x_min) / gauge_width,
                (needle_y_max - gauge_y_min) / gauge_height,
            ]
            image = cropped_image
        else:
            # If not box_only, return a default bbox covering the full image.
            needle_bbox = [0, 0, 1, 1]

        if self.transform:
            sample = {'image': image, 'bbox': needle_bbox}
            # Transform the sample
            transformed = self.transform(sample)

            # Extract image and bounding box
            image = transformed['image']
            needle_bbox = transformed['bbox']
        # print(needle_bbox)
        image = transforms.ToTensor()(image)
        return (
            image,
            torch.tensor(needle_bbox, dtype=torch.float32),
            torch.tensor([rotation], dtype=torch.float32),
        )
