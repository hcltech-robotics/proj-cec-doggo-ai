# File: flask_app/app.py
from flask import Flask, request, jsonify
from functools import wraps
import argparse
import base64
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
import custom_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Gauge Reader AI Service")
    parser.add_argument(
        "--token", type=str, required=True, 
        help="Static API token for authentication"
    )
    parser.add_argument(
        "--detector-model", dest="detector_model", required=True,
        help="Path to the detector model file (.pth)"
    )
    parser.add_argument(
        "--reader-model", dest="reader_model", required=True,
        help="Path to the reader model file (.pth)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Flask host interface"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Flask port"
    )
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()
API_TOKEN = args.token

# Flask application
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Authentication decorator
def require_token(f):
    @wraps(f)
    def decorated(*f_args, **f_kwargs):
        auth_header = request.headers.get('Authorization', '')
        if auth_header != f'Bearer {API_TOKEN}':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*f_args, **f_kwargs)
    return decorated

# Load detector model
_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=3)
_in_features = _detector.roi_heads.box_predictor.cls_score.in_features
_detector.roi_heads.box_predictor = FastRCNNPredictor(_in_features, 3)
_detector.load_state_dict(
    torch.load(args.detector_model, map_location=device, weights_only=True)
)
_detector.to(device).eval()
_detector_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load reader model
_reader = torch.jit.load(args.reader_model, map_location=device)
_reader.to(device).eval()
_reader_transform = transforms.Compose([
    custom_transform.CLAHEPreprocess(),
    custom_transform.ResizeWithPaddingAndBBox(),
])

# Utility to decode base64 image
def decode_image(b64_str):
    img_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def _sanitize_and_crop(img: np.ndarray, bbox_norm, min_size: int = 2):
    """
    bbox_norm = [x1_norm, y1_norm, x2_norm, y2_norm], values in [0,1]
    Returns (crop, [x1_px, y1_px, x2_px, y2_px]) with safe ints.
    Raises ValueError on invalid input.
    """
    H, W = img.shape[:2]

    # convert & clamp normalized bbox to [0,1]
    bb = np.asarray(bbox_norm, dtype=np.float32)
    if bb.shape != (4,):
        raise ValueError(f"bbox must be length-4, got shape {bb.shape}")

    bb = np.clip(bb, 0.0, 1.0)
    x1f, y1f, x2f, y2f = bb.tolist()

    # ensure proper ordering
    if x2f < x1f: x1f, x2f = x2f, x1f
    if y2f < y1f: y1f, y2f = y2f, y1f

    # convert to pixel coords
    x1 = int(round(x1f * W))
    y1 = int(round(y1f * H))
    x2 = int(round(x2f * W))
    y2 = int(round(y2f * H))

    # clamp to image bounds
    x1 = max(0, min(x1, W))
    x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H))
    y2 = max(0, min(y2, H))

    # enforce minimal, non-empty crop
    if x2 - x1 < min_size or y2 - y1 < min_size:
        raise ValueError(f"Empty/too-small crop after clamp: {(x1,y1,x2,y2)} in {W}x{H}")

    crop = img[y1:y2, x1:x2]
    return crop, [x1, y1, x2, y2]


@app.route('/detect', methods=['POST'])
@require_token
def detect():
    data = request.get_json()
    img = decode_image(data['image'])
    tensor = _detector_transform(img).to(device)
    with torch.no_grad():
        out = _detector([tensor])[0]
    boxes = out['boxes'].cpu().numpy().tolist()
    scores = out['scores'].cpu().numpy().tolist()
    labels = out['labels'].cpu().numpy().tolist()
    best = {'gauge': {'bbox': None, 'score': 0}, 'needle': {'bbox': None, 'score': 0}}
    for box, score, label in zip(boxes, scores, labels):
        if label == 1 and score > best['gauge']['score']:
            best['gauge'] = {'bbox': box, 'score': score}
        if label == 2 and score > best['needle']['score']:
            best['needle'] = {'bbox': box, 'score': score}
    return jsonify(best)

@app.route('/read', methods=['POST'])
@require_token
def read():
    data = request.get_json()
    img = decode_image(data['image'])  # expect HxWxC RGB/uint8
    bbox_norm = data['bbox']           # normalized [x1,y1,x2,y2] floats

    try:
        crop, (x1, y1, x2, y2) = _sanitize_and_crop(img, bbox_norm, min_size=2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Safe renormalization inside the crop’s frame
    ch, cw = crop.shape[:2]
    bbox_in_crop_norm = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # full crop by default
    # If you actually intended to keep the original bbox expressed relative to the crop:
    # bbox_in_crop_norm = np.array([0,0,cw,ch], np.float32) / np.array([cw, ch, cw, ch], np.float32)

    sample = {
        "image": crop,
        "bbox": bbox_in_crop_norm,
    }

    transformed = _reader_transform(sample)
    gauge_img = transformed["image"]
    bbox_norm_t = transformed["bbox"]

    if gauge_img is None or getattr(gauge_img, "size", 0) == 0:
        return jsonify({"error": "Transform produced empty image"}), 500

    tensor = transforms.ToTensor()(gauge_img).unsqueeze(0).to(device)
    bbox_t = torch.tensor(bbox_norm_t, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = _reader(tensor, bbox_t)

    return jsonify({"reading": float(output.item())})

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
