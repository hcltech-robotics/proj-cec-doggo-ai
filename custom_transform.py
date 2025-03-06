import numpy as np
import cv2
from PIL import Image, ImageOps

class Noise:
    def __init__(self, poisson_noise=True, blur_kernel=(5,5)):
        self.poisson_noise = poisson_noise
        self.blur_kernel = blur_kernel

    def add_poisson_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))  # Adjust to power of 2
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy_image

    def __call__(self, sample):
        if isinstance(sample, dict):
            img = sample['image']
        elif isinstance(sample, Image.Image):
            img = sample

        opencv_image = np.array(img)
        
         # Blur
        if self.blur_kernel:
            opencv_image = cv2.GaussianBlur(opencv_image, self.blur_kernel, 0)
        # Noise
        if self.poisson_noise:
            opencv_image = self.add_poisson_noise(opencv_image)

        img = Image.fromarray(opencv_image)

        if isinstance(sample, dict):
            result = sample.copy()
            result['image'] = img
        elif isinstance(sample, Image.Image):
            result = img

        return result

class CLAHEPreprocess:
    def __init__(self, clip_limit=5.0, tile_grid_size=(10, 10), min_area=15000, poisson_noise=False, blur_kernel=None):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.min_area = min_area
        self.blur_kernel = blur_kernel
        self.poisson_noise = poisson_noise

    def add_poisson_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))  # Adjust to power of 2
        noisy = np.random.poisson(image * vals) / float(vals)
        noisy_image = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy_image

    def __call__(self, sample):
        img = sample['image']
        bbox = sample['bbox']

        # Convert PIL to OpenCV (NumPy array)
        opencv_image = np.array(img)


        # Blur
        if self.blur_kernel:
            opencv_image = cv2.GaussianBlur(opencv_image, self.blur_kernel, 0)
        # Noise
        if self.poisson_noise:
            opencv_image = self.add_poisson_noise(opencv_image)

        # Convert image to grayscale (handling RGB/RGBA cases)
        if opencv_image.shape[-1] == 4:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2GRAY)
        elif opencv_image.shape[-1] == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced_image = clahe.apply(opencv_image)

        # Apply thresholding
        _, threshold_image = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY)

        # Morphological closing
        kernel = np.ones((7, 7), np.uint8)
        closed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)

        # Find and filter contours
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        # Draw filtered contours onto a new mask
        result_image = np.zeros_like(opencv_image)
        cv2.drawContours(result_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

        # Apply the mask
        masked_threshold = cv2.bitwise_and(threshold_image, threshold_image, mask=result_image)

        # Convert back to PIL Image
        img = Image.fromarray(masked_threshold)

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

        orig_bbox = bbox * np.array(
            [original_width, original_height, original_width, original_height]
        )
        orig_bbox = orig_bbox.astype(int)

        # Scale the bounding box
        bbox = [
            orig_bbox[0] * scale_x,
            orig_bbox[1] * scale_y,
            orig_bbox[2] * scale_x,
            orig_bbox[3] * scale_y,
        ]
        # Calculate padding to center the image
        width_padding = max(0, (self.target_size[0] - new_width))
        height_padding = max(0, (self.target_size[1] - new_height))

        # Apply padding
        img = ImageOps.expand(
            img, border=(0, 0, width_padding, height_padding), fill=self.padding_color
        )

        # Adjust the bounding box based on the scaling
        scaled_bbox = [
            bbox[0] / new_width,
            bbox[1] / new_height,
            bbox[2] / new_width,
            bbox[3] / new_height,
        ]  # xmin  # ymin  # xmax  # ymax

        return {'image': img, 'bbox': scaled_bbox}
