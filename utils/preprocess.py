import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def clahe_green_channel(image):
    image = np.array(image)

    green_channel = image[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green_channel)

    enhanced = cv2.merge([enhanced, enhanced, enhanced])

    return Image.fromarray(enhanced)


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])