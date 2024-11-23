import argparse
import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import UnetPlusPlus

# Argument parsing
parser = argparse.ArgumentParser(description="Inference script for image segmentation")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the segmented image")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
args = parser.parse_args()

# Transformation for the input image
def get_transform():
    return Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# Function to map mask to colors
def mask_to_rgb(mask):
    color_dict = {
        0: (0, 0, 0),       # Background
        1: (255, 0, 0),     # Class 1 - Red
        2: (0, 255, 0),     # Class 2 - Green
    }
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_dict.items():
        rgb_mask[mask == cls] = color
    return rgb_mask

# Load the checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnetPlusPlus(
    encoder_name="resnet34", 
    encoder_weights=None, 
    in_channels=3, 
    classes=3
)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load and preprocess the image
image_path = args.image_path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Input image not found at {image_path}")

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
original_size = (image.shape[1], image.shape[0])  # (width, height)

transform = get_transform()
transformed = transform(image=image)
input_tensor = transformed["image"].unsqueeze(0).to(device)

# Perform inference
with torch.no_grad():
    output = model(input_tensor)
    output = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()

# Resize mask to original size and convert to RGB
output_mask = cv2.resize(output, original_size, interpolation=cv2.INTER_NEAREST)
output_rgb = mask_to_rgb(output_mask)

# Save the output
output_path = args.output_path
output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, output_rgb)
print(f"Segmented image saved at {output_path}")

