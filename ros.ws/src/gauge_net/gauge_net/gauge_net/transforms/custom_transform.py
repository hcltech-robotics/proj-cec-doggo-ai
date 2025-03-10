import cv2
import numpy as np
from PIL import Image, ImageOps


class CLAHEPreprocess:
    def __init__(self, clip_limit=5.0, tile_grid_size=(10, 10), min_area_ratio=0.01, threshold_scaling=0.8):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.min_area_ratio = min_area_ratio  # Dynamic min_area based on image size
        self.threshold_scaling = threshold_scaling  # Scaling factor for adaptive thresholding

    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']

        # Convert PIL to OpenCV (NumPy array)
        opencv_image = np.array(img)

        # Convert to grayscale (handling RGB/RGBA cases)
        if opencv_image.shape[-1] == 4:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2GRAY)
        elif opencv_image.shape[-1] == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)

        # Step 1: Apply CLAHE to enhance contrast and reduce shadows
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_image = clahe.apply(opencv_image)

        # Step 2: Adaptive thresholding for needle extraction (dark regions)
        median_intensity = np.median(enhanced_image)
        needle_threshold_value = int(median_intensity * self.threshold_scaling)
        _, needle_mask = cv2.threshold(enhanced_image, needle_threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Step 3: Detect the gauge face (bright regions) to remove outer background
        _, gauge_mask = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Find the largest contour 
        contours, _ = cv2.findContours(gauge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            face_mask = np.zeros_like(gauge_mask)
            cv2.drawContours(face_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        else:
            face_mask = gauge_mask  # Fallback if no contours are detected

        # Step 5: Apply the final mask to keep only the gauge face and needle
        final_mask = cv2.bitwise_and(needle_mask, face_mask)

        # Convert back to PIL Image
        img = Image.fromarray(final_mask)

        return {'image': img, 'bbox': bbox}



class ResizeWithPaddingAndBBox:

    def __init__(self, target_size=(512, 512), padding_color=0):
        self.target_size = target_size
        self.padding_color = padding_color

    def __call__(self, sample):
        img, bbox = sample['image'], sample['bbox']
        # Step 1: Resize the image while maintaining the aspect ratio
        original_img = img.copy()
        img = ImageOps.contain(img, self.target_size)
        original_width, original_height = original_img.size
        new_width, new_height = img.size

        # Calculate precise scaling factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height

        # Scale the bounding box from normalized to pixel coordinates
        orig_bbox = bbox * np.array(
            [original_width, original_height, original_width, original_height]
        )
        orig_bbox = orig_bbox.astype(int)

        # Scale the bounding box
        scaled_bbox_pixels = [
            orig_bbox[0] * scale_x,
            orig_bbox[1] * scale_y,
            orig_bbox[2] * scale_x,
            orig_bbox[3] * scale_y,
        ]

        # Calculate padding to fill target size (right and bottom only)
        width_padding = max(0, (self.target_size[0] - new_width))
        height_padding = max(0, (self.target_size[1] - new_height))

        # Apply padding to right and bottom only
        img = ImageOps.expand(
            img, border=(0, 0, width_padding, height_padding), fill=self.padding_color
        )

        # Get final image dimensions (after padding)
        final_width, final_height = img.size

        # Normalize the bounding box based on the final padded image dimensions
        scaled_bbox = [
            scaled_bbox_pixels[0] / final_width,
            scaled_bbox_pixels[1] / final_height,
            scaled_bbox_pixels[2] / final_width,
            scaled_bbox_pixels[3] / final_height,
        ]

        return {'image': img, 'bbox': scaled_bbox}
