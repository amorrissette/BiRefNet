#!/usr/bin/env python3
"""
Utility script to convert multiple grayscale masks to a combined RGB mask.

This script takes multiple grayscale mask images (each representing a different
channel/feature) and combines them into a single RGB mask image where each
input mask is assigned a specific color.

Usage:
    python convert_masks_to_rgb.py --output combined_mask.png \
        --mask1 foreground_mask.png --color1 255,0,0 \
        --mask2 edge_mask.png --color2 0,255,0 \
        --mask3 detail_mask.png --color3 0,0,255
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_color(color_str):
    """Parse a comma-separated RGB color string into a tuple of integers."""
    try:
        r, g, b = map(int, color_str.split(','))
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
        return (r, g, b)
    except (ValueError, AssertionError):
        raise ValueError(f"Invalid color format: {color_str}. Expected format: r,g,b (e.g., 255,0,0)")

def main():
    parser = argparse.ArgumentParser(description="Convert multiple grayscale masks to a combined RGB mask")
    parser.add_argument('--output', required=True, help='Path to save combined RGB mask')
    
    # Define up to 5 mask-color pairs
    for i in range(1, 6):
        parser.add_argument(f'--mask{i}', help=f'Path to grayscale mask {i}')
        parser.add_argument(f'--color{i}', help=f'RGB color for mask {i} (comma-separated, e.g., 255,0,0)')
    
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Threshold for binary mask (0.0-1.0)')
    parser.add_argument('--visualize', action='store_true', 
                        help='Show a visualization of the result')
    
    args = parser.parse_args()
    
    # Collect mask-color pairs
    mask_color_pairs = []
    for i in range(1, 6):
        mask_path = getattr(args, f'mask{i}', None)
        color_str = getattr(args, f'color{i}', None)
        
        if mask_path and color_str:
            color = parse_color(color_str)
            mask_color_pairs.append((mask_path, color))
    
    if not mask_color_pairs:
        parser.error("At least one mask-color pair is required (--mask1 and --color1)")
    
    # Load first mask to get dimensions
    first_mask_path = mask_color_pairs[0][0]
    first_mask = np.array(Image.open(first_mask_path).convert('L'))
    height, width = first_mask.shape
    
    # Create empty RGB image
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process each mask-color pair
    for mask_path, color in mask_color_pairs:
        # Load grayscale mask
        grayscale_mask = np.array(Image.open(mask_path).convert('L'))
        
        # Resize if dimensions don't match
        if grayscale_mask.shape != (height, width):
            print(f"Warning: Resizing mask {mask_path} to match dimensions of first mask")
            temp_img = Image.fromarray(grayscale_mask)
            temp_img = temp_img.resize((width, height), Image.NEAREST)
            grayscale_mask = np.array(temp_img)
        
        # Normalize to 0-1 if not already
        if grayscale_mask.max() > 1:
            grayscale_mask = grayscale_mask / 255.0
        
        # Apply threshold to binarize mask
        binary_mask = grayscale_mask >= args.threshold
        
        # Set RGB values where mask is active
        for i, channel_value in enumerate(color):
            rgb_mask[binary_mask, i] = channel_value
    
    # Save RGB mask
    rgb_image = Image.fromarray(rgb_mask)
    rgb_image.save(args.output)
    print(f"Saved RGB mask to {args.output}")
    
    # Visualize if requested
    if args.visualize:
        plt.figure(figsize=(12, 6))
        
        # Plot input masks
        num_inputs = len(mask_color_pairs)
        for i, (mask_path, color) in enumerate(mask_color_pairs):
            plt.subplot(2, num_inputs, i + 1)
            mask = np.array(Image.open(mask_path).convert('L'))
            if mask.max() > 1:
                mask = mask / 255.0
            plt.imshow(mask, cmap='gray')
            plt.title(os.path.basename(mask_path))
            plt.axis('off')
        
        # Plot combined RGB mask
        plt.subplot(2, 1, 2)
        plt.imshow(rgb_mask)
        plt.title('Combined RGB Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()