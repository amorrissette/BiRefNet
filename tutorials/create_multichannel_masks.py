#!/usr/bin/env python3
"""
Example script that converts RGB label images to multi-channel labels
and demonstrates how to work with the color-to-channel mapping.

This script requires PIL, numpy, and matplotlib.
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_color_map(color_channel_map):
    """
    Create a colormap for visualizing multi-channel segmentation maps.
    """
    if not color_channel_map:
        return None
        
    # Extract colors and channel indices
    colors = []
    for (r, g, b), channel in sorted(color_channel_map.items(), key=lambda x: x[1]):
        colors.append((r/255, g/255, b/255))
    
    # Add black as the first color (background)
    colors.insert(0, (0, 0, 0))
    
    return LinearSegmentedColormap.from_list("segmentation_cmap", colors, N=len(colors))

def rgb_to_multi_channel(rgb_image, color_channel_map, num_channels):
    """
    Convert an RGB image to a multi-channel mask based on the color mapping.
    
    Args:
        rgb_image: RGB PIL Image
        color_channel_map: Dict mapping RGB tuples to channel indices
        num_channels: Number of output channels
        
    Returns:
        numpy array with shape [num_channels, H, W]
    """
    # Convert PIL image to numpy array
    rgb_array = np.array(rgb_image)
    h, w, _ = rgb_array.shape
    
    # Create output array
    multi_channel = np.zeros((num_channels, h, w), dtype=np.float32)
    
    # Process each color in the mapping
    for (r, g, b), channel_idx in color_channel_map.items():
        if channel_idx >= num_channels:
            continue
            
        # Create a mask where all RGB channels match the target color
        color_mask = np.all(rgb_array == np.array([r, g, b]), axis=2)
        
        # Set the corresponding channel to 1 where the color matches
        multi_channel[channel_idx][color_mask] = 1.0
        
    return multi_channel

def visualize_channels(multi_channel_array, output_path=None):
    """
    Visualize each channel in the multi-channel array.
    
    Args:
        multi_channel_array: numpy array with shape [num_channels, H, W]
        output_path: Optional path to save the visualization
    """
    num_channels = multi_channel_array.shape[0]
    
    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 4, 4))
    
    if num_channels == 1:
        axes = [axes]  # Make iterable for single channel case
        
    for i, ax in enumerate(axes):
        ax.imshow(multi_channel_array[i], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Channel {i}")
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

def visualize_combined(multi_channel_array, color_channel_map, output_path=None):
    """
    Create a combined visualization of all channels with different colors.
    
    Args:
        multi_channel_array: numpy array with shape [num_channels, H, W]
        color_channel_map: Dict mapping RGB tuples to channel indices
        output_path: Optional path to save the visualization
    """
    num_channels = multi_channel_array.shape[0]
    h, w = multi_channel_array.shape[1:]
    
    # Create RGB visualization
    rgb_vis = np.zeros((h, w, 3), dtype=np.float32)
    
    # Add each channel with its corresponding color
    for (r, g, b), channel_idx in color_channel_map.items():
        if channel_idx >= num_channels:
            continue
        
        # Get the channel mask
        mask = multi_channel_array[channel_idx]
        
        # Add color to the RGB visualization where mask is active
        rgb_vis[mask > 0.5] = np.array([r, g, b]) / 255.0
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_vis)
    plt.title("Combined Channels")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved combined visualization to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Convert RGB label images to multi-channel segmentation masks")
    parser.add_argument('--input', required=True, help='Path to input RGB label image')
    parser.add_argument('--output_dir', default='./output', help='Directory to save outputs')
    args = parser.parse_args()
    
    # Example color channel mapping
    color_channel_map = {
        (255, 0, 0): 0,    # Red pixels go to channel 0
        (0, 255, 0): 1,    # Green pixels go to channel 1
        (0, 0, 255): 2,    # Blue pixels go to channel 2
    }
    num_channels = 3
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the input image
    image = Image.open(args.input).convert('RGB')
    
    # Convert to multi-channel
    multi_channel = rgb_to_multi_channel(image, color_channel_map, num_channels)
    
    # Visualize individual channels
    visualize_channels(multi_channel, os.path.join(args.output_dir, 'channels.png'))
    
    # Visualize combined channels
    visualize_combined(multi_channel, color_channel_map, os.path.join(args.output_dir, 'combined.png'))
    
    print(f"Processed {args.input} into {num_channels} channels")
    
    # For demonstration: show how to access individual channels
    print("\nExample pixel values:")
    h, w = multi_channel.shape[1:]
    center_y, center_x = h // 2, w // 2
    for i in range(num_channels):
        print(f"Channel {i} at center pixel: {multi_channel[i, center_y, center_x]}")

if __name__ == "__main__":
    main()