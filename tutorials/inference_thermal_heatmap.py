#!/usr/bin/env python
# Script for running inference with a trained thermal heatmap model

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from matplotlib import pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import generate_heatmap_from_json
from models.birefnet import BiRefNet
from utils import save_tensor_img
from config import Config

def run_inference(model_path, image_path, json_path=None, output_dir='output'):
    """
    Run inference with a trained model on a thermal image
    
    Args:
        model_path: Path to the trained model weights (.pth file)
        image_path: Path to the thermal image (.png file)
        json_path: Optional path to thermal points JSON file
        output_dir: Directory to save output results
    """
    # Initialize config
    config = Config()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = BiRefNet(bb_pretrained=False)
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()
    model.half()  # Use half precision for inference
    
    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    
    # Convert to RGB (model expects 3 channels, even though grayscale may be used)
    if config.grayscale_input:
        image_array = np.array(image.convert('L'))
        image_rgb = np.stack([image_array, image_array, image_array], axis=2)
        image = Image.fromarray(image_rgb)
    
    # Resize to be divisible by 32 (required by model)
    orig_size = image.size
    size_div_32 = (int(orig_size[0] // 32 * 32), int(orig_size[1] // 32 * 32))
    if image.size != size_div_32:
        image = image.resize(size_div_32)
    
    # Convert to tensor and normalize
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(config.device).half()
    
    # Generate prediction
    with torch.no_grad():
        scaled_preds = model(input_tensor)[-1].sigmoid()
    
    # Resize prediction back to original size
    res = torch.nn.functional.interpolate(
        scaled_preds[0].unsqueeze(0),
        size=orig_size[::-1],  # PIL Image size is (width, height) but interpolate expects (height, width)
        mode='bilinear',
        align_corners=True
    )
    
    # Save prediction
    output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
    save_tensor_img(res, output_path)
    print(f"Saved prediction to {output_path}")
    
    # Generate visualization
    pred_np = res[0][0].cpu().numpy()
    
    # Create a visualization with various components
    plt.figure(figsize=(15, 6))
    
    # 1. Original thermal image
    plt.subplot(131)
    plt.imshow(np.array(Image.open(image_path)), cmap='gray')
    plt.title("Original Thermal Image")
    plt.axis('off')
    
    # 2. Ground truth heatmap (if JSON is provided)
    if json_path and os.path.exists(json_path):
        plt.subplot(132)
        heatmap = generate_heatmap_from_json(json_path)
        plt.imshow(np.array(heatmap), cmap='Blues')
        plt.title("Ground Truth Heatmap")
        plt.axis('off')
    
    # 3. Predicted heatmap
    plt.subplot(133)
    plt.imshow(pred_np, cmap='hot')
    plt.title("Predicted Heatmap")
    plt.axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f"visualization_{os.path.basename(image_path).replace('.png', '.jpg')}")
    plt.savefig(vis_path)
    print(f"Saved visualization to {vis_path}")
    
    # Create overlay of prediction on original image
    original_img = np.array(Image.open(image_path).convert('RGB'))
    heatmap_color = cv2.applyColorMap((pred_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (original_img.shape[1], original_img.shape[0]))
    
    # Blend images
    alpha = 0.6  # Transparency factor
    overlay = cv2.addWeighted(original_img, 1-alpha, heatmap_color, alpha, 0)
    
    # Save overlay
    overlay_path = os.path.join(output_dir, f"overlay_{os.path.basename(image_path)}")
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved overlay image to {overlay_path}")
    
    return output_path, vis_path, overlay_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with a trained thermal heatmap model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pth file)')
    parser.add_argument('--image', type=str, required=True, help='Path to thermal image (.png file)')
    parser.add_argument('--json', type=str, help='Optional path to thermal points JSON file')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(args.model, args.image, args.json, args.output)