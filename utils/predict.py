import torch
import torch.nn as nn
import timm
import numpy as np
from utils.preprocess import transform, clahe_green_channel
from class_names import CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create the base ResNet model
model = timm.create_model('resnet50', pretrained=False)

# 2. Reconstruct the custom classifier head
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 5)
)

# 3. Load the weights
model.load_state_dict(torch.load("resnet_best_final.pth", map_location=device))
model.to(device)
model.eval()

# --- 🚀 ADD YOUR OPTIMIZED THRESHOLDS HERE ---
# Replace these with the exact numbers printed by your training notebook
OPTIMAL_THRESHOLDS = [0.5, 1.5, 2.5, 3.5] 

def get_class_from_score(score, thresholds):
    """Converts a continuous scalar score to a class index (0-4) using thresholds."""
    pred = 0
    for i, thr in enumerate(thresholds):
        if score > thr:
            pred = i + 1
    return pred

def predict_image(image):
    image = clahe_green_channel(image)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # --- TEST TIME AUGMENTATION (TTA) ---
        # 1. Original Image
        logits_orig = model(tensor)
        # 2. Horizontal Flip
        logits_hflip = model(torch.flip(tensor, [3]))
        # 3. Vertical Flip
        logits_vflip = model(torch.flip(tensor, [2]))

        # Average the logits
        avg_logits = (logits_orig + logits_hflip + logits_vflip) / 3.0
        
        # Get Probabilities
        probs = torch.softmax(avg_logits, dim=1)

        # --- CONTINUOUS SCORE CALCULATION ---
        # Multiply probabilities by [0, 1, 2, 3, 4] and sum them up
        weights = torch.arange(5, dtype=torch.float32, device=device)
        continuous_score = (probs * weights).sum(dim=1).item()

    # --- APPLY OPTIMIZED THRESHOLDS ---
    pred_class = get_class_from_score(continuous_score, OPTIMAL_THRESHOLDS)

    # For the UI confidence, we grab the probability of the final chosen class
    confidence = probs[0][pred_class].item() * 100

    return {
        "class_index": pred_class,
        "class_name": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "probabilities": probs.squeeze().cpu().numpy(),
        "continuous_score": continuous_score
    }