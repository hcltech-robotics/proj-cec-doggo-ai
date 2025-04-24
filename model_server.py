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
    img = decode_image(data['image'])
    bbox = data['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    crop = img[y1:y2, x1:x2]
    sample = {
        'image': crop,
        'bbox': np.array([x1, y1, x2, y2]) / np.array([crop.shape[1], crop.shape[0], crop.shape[1], crop.shape[0]])
    }
    transformed = _reader_transform(sample)
    gauge_img = transformed['image']
    bbox_norm = transformed['bbox']
    tensor = transforms.ToTensor()(gauge_img).unsqueeze(0).to(device)
    bbox_t = torch.tensor(bbox_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = _reader(tensor, bbox_t)
    return jsonify({'reading': float(output.item())})

if __name__ == '__main__':
    app.run(host=args.host, port=args.port)
