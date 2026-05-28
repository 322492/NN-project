import torch
from PIL import Image
from torch.nn.functional import sigmoid
from torchvision import transforms

class SlidingWindow:
    def __init__(self, cnn, window_sizes, overlap_ratio, threshold=0.7, crop_size=128, device="cpu"):
        self.cnn = cnn
        self.cnn.eval()

        self.window_sizes = window_sizes
        self.overlap_ratio = overlap_ratio
        self.threshold = threshold
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.device = device

    def _generate_windows(self, image_width, image_height):
        windows = []

        for window_size in self.window_sizes:
            stride = int(window_size * (1- self.overlap_ratio))
            stride = max(1, stride)

            for x1 in range(0, image_width - window_size+1, stride):
                for y1 in range(0, image_height - window_size+1, stride):
                    x2 = x1 + window_size
                    y2 = y1 + window_size
                    windows.append(([x1, y1, x2, y2]))

        return windows

    @torch.no_grad()
    def predict_window(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        windows = self._generate_windows(image_width, image_height)

        boxes = []
        confidence = []

        for window in windows:
            x1, y1, x2, y2 = window

            crop = image.crop((x1, y1, x2, y2))
            crop = self.transform(crop)
            crop = crop.unsqueeze(0).to(self.device)

            logits = self.cnn(crop) # [1,1]
            probs = torch.sigmoid(logits).item()    #liczba

            if probs >= self.threshold:
                boxes.append(window)
                confidence.append(probs)

        return boxes, confidence