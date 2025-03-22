import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional


class RandomHomographyTransform(torch.nn.Module):

    def __init__(self, max_warp=4):
        """
        max_warp determines how far corners can be perturbed (in pixels).
        """
        self.max_warp = max_warp
        super().__init__()

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = functional.to_pil_image(image)  # Convert tensor to PIL Image
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        W, H = image.size  # Get width and height of the image

        # Define source points (corners of the original image)
        src_pts = [
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1],
        ]

        # Create randomly perturbed destination points
        dst_pts = []
        for (x, y) in src_pts:
            dx = random.randint(-self.max_warp, self.max_warp)
            dy = random.randint(-self.max_warp, self.max_warp)
            nx = min(max(x + dx, 0), W - 1)
            ny = min(max(y + dy, 0), H - 1)
            dst_pts.append([nx, ny])

        # Apply the perspective transform and specify fill=0 for black background
        transformed_image = functional.perspective(
            image,
            startpoints=src_pts,
            endpoints=dst_pts,
            interpolation=functional.InterpolationMode.BILINEAR,
            fill=0  # Ensure areas outside the original image are padded with black
        )

        return transformed_image
