import os
import json
import argparse

import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

def load_model(model_path, device):
    """
    Loads the TorchScript model.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

def get_transform():
    """
    Returns the same image transforms as during training (without resize).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def draw_boxes(image, boxes, labels, scores, score_threshold=0.5):
    """
    Draws bounding boxes and labels on an image.
    Only boxes with a score above the threshold are drawn.
    """
    draw = ImageDraw.Draw(image)
    # Try to use a truetype font; fall back to default if unavailable
    font = ImageFont.load_default(size = 20)

    # Mapping from numerical labels to human-readable strings
    label_map = {1: "gauge", 2: "gauge_needle"}
    
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x_min, y_min, x_max, y_max = box
        # Draw rectangle for the bounding box
        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=2)
        # Prepare text (label and score)
        label_name = label_map.get(label, str(label))
        text = f"{label_name}: {score:.2f}"
        # Compute text dimensions using draw.textbbox
        bbox = draw.textbbox((x_min, y_min), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Ensure text background does not go above the image
        text_y = max(0, y_min - text_height)
        # Draw filled rectangle behind the text for readability
        draw.rectangle([x_min, text_y, x_min + text_width, text_y + text_height], fill="red")
        # Draw the text on top
        draw.text((x_min, text_y), text, fill="white", font=font)
    return image



def run_inference(input_dir, model, device, score_threshold):
    """
    Runs inference on all images in the input directory.
    Returns a dictionary with detections for each image.
    """
    transform = get_transform()
    results = {}

    # Consider common image extensions
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        # Open image and make a copy for drawing
        image = Image.open(img_path).convert("RGB")
        original_image = image.copy()

        # Apply transforms for model input
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Run the model
        with torch.no_grad():
            prediction = model([img_tensor[0]])[1][0]
        # Convert outputs to lists
        boxes = prediction["boxes"].cpu().numpy().tolist()
        labels = prediction["labels"].cpu().numpy().tolist()
        scores = prediction["scores"].cpu().numpy().tolist()

        # Filter detections using the score threshold
        filtered_boxes, filtered_labels, filtered_scores = [], [], []
        for box, label, score in zip(boxes, labels, scores):
            if score >= score_threshold:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)

        # Save detection results for the image
        results[img_file] = []
        label_map = {1: "gauge", 2: "gauge_needle"}
        for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
            results[img_file].append({
                "bbox": box,
                "label": label_map.get(label, str(label)),
                "score": score
            })

        # Draw the bounding boxes and labels on the image copy
        drawn_image = draw_boxes(original_image, filtered_boxes, filtered_labels, filtered_scores, score_threshold)
        # Save the image with drawn boxes to an output folder
        yield img_file, drawn_image, results[img_file]

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(args.model_path, device)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    overall_results = {}
    # Process each image
    for img_file, drawn_image, detections in run_inference(args.input_dir, model, device, args.score_threshold):
        # Save drawn image
        output_img_path = os.path.join(args.output_dir, img_file)
        drawn_image.save(output_img_path)
        print(f"Processed and saved: {output_img_path}")
        overall_results[img_file] = detections

    # Save detection results to JSON file
    json_file_path = os.path.join(args.output_dir, args.output_json)
    with open(json_file_path, "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"Detection results saved to {json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a TorchScript Faster R-CNN model and output bounding boxes.")
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Directory containing input images.")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the TorchScript model file (e.g., gauge_detect.pt).")
    parser.add_argument('--output_json', type=str, default="detections.json",
                        help="Path to the output JSON file to save detections.")
    parser.add_argument('--output_dir', type=str, default="output_images",
                        help="Directory to save images with drawn bounding boxes.")
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help="Score threshold for filtering predictions.")
    args = parser.parse_args()
    
    main(args)


#python3 gauge_detect_test.py --input_dir /home/ifodor/Documents/Projects/omniverse-examples/extracted_images --model_path checkpoints/gauge_detect.pt --score_threshold 0.999 --output_dir /home/ifodor/Documents/Projects/omniverse-examples/labeled_images
