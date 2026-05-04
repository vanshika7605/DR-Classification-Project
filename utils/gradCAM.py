import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.predict import model, device
from utils.preprocess import transform, clahe_green_channel

def generate_gradcam(image, target_class_index=None):
    enhanced_img = clahe_green_channel(image)
    input_tensor = transform(enhanced_img).unsqueeze(0).to(device)

    bg_img = enhanced_img.resize((512, 512))
    rgb_img = np.array(bg_img, dtype=np.float32) / 255.0

    # TARGET LAYER CHANGE: For ResNet, the best layer for Grad-CAM is the final block
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        
        targets = None
        if target_class_index is not None:
            targets = [ClassifierOutputTarget(target_class_index)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return Image.fromarray(visualization)