# Multi-Channel Segmentation Tutorial

BiRefNet now supports generating multiple output segmentation maps from a single RGB image. This tutorial demonstrates how to configure and use this feature.

## Overview

With multi-channel segmentation support, you can:

- Train a model to predict multiple different segmentation maps simultaneously
- Use RGB label images where different colors correspond to different output channels
- Convert RGB color values in label images to separate binary masks 

## Configuration

To use multi-channel segmentation:

1. Set the number of output channels in your config
2. Enable RGB label images
3. Define a color-to-channel mapping

```python
# In your config.py or when initializing Config()
config = Config()

# Set the number of output channels (default is 1)
config.num_output_channels = 3

# Enable RGB label images (instead of grayscale)
config.rgb_labels = True

# Define the mapping from RGB colors to channel indices (0-based)
config.color_channel_map = {
    (255, 0, 0): 0,    # Red pixels go to channel 0 (foreground)
    (0, 255, 0): 1,    # Green pixels go to channel 1 (edges)
    (0, 0, 255): 2,    # Blue pixels go to channel 2 (other feature)
}

# Set color tolerance for approximate matching (0 for exact matching)
config.color_tolerance = 10    # Colors within 10 units of target RGB values will match
```

## Dataset Preparation

Your dataset should contain:

1. Input images in the standard format
2. RGB label images where different colors correspond to different features

For example:
- Input RGB image: `dataset/training/im/image1.jpg`
- RGB label image: `dataset/training/gt/image1.png` (containing red, green, and blue pixels)

## Model Output

The model will output a tensor with shape `[batch_size, num_output_channels, height, width]` where:

- Each channel contains a segmentation map for one feature
- For the example above:
  - Channel 0: Segmentation map for red pixels (foreground)
  - Channel 1: Segmentation map for green pixels (edges)
  - Channel 2: Segmentation map for blue pixels (other feature)

## Inference Example

```python
import torch
from PIL import Image
import numpy as np
from models.birefnet import BiRefNet

# Load your model
model = BiRefNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Process an image
image = Image.open('input.jpg').convert('RGB')
# Transform image as required
# ...

# Get model prediction
with torch.no_grad():
    outputs = model(image_tensor.unsqueeze(0))
    
# outputs will have shape [1, num_output_channels, H, W]
# Extract individual segmentation maps
foreground_mask = outputs[0, 0].cpu().numpy()  # Channel 0
edge_mask = outputs[0, 1].cpu().numpy()        # Channel 1
other_feature = outputs[0, 2].cpu().numpy()    # Channel 2

# Visualize or save results as needed
```

## Training Tips

1. **Data preparation**: Ensure your RGB label images are correctly colored with the colors defined in your `color_channel_map`.

2. **Configuration**: Make sure `num_output_channels` matches the number of channels you want to predict.

3. **Loss function**: The loss function has been updated to support both multi-channel and single-channel ground truth.

4. **Performance**: Computing multiple segmentation maps may require more GPU memory and training time.

## Approximate Color Matching

The system supports approximate color matching to handle small variations in RGB values:

- Set `config.color_tolerance` to a positive integer (default: 10) to enable approximate matching
- Colors within the specified distance will be assigned to the corresponding channel
- This makes the system more robust to JPEG compression artifacts and anti-aliasing effects
- Use a smaller tolerance (or 0) for exact color matching when precision is required

For example, with `color_tolerance = 10`:
- RGB value (255, 0, 0) will match pixels with R between 245-255, G between 0-10, and B between 0-10
- This helps with slightly varying colors at object boundaries or from image compression

You can adjust this value based on your dataset characteristics - higher for more tolerance, lower for stricter matching.

## Compatibility

This feature is fully compatible with all of BiRefNet's existing functionality, including:

- All backbone architectures
- Both single-scale and multi-scale supervision
- Training with background color synthesis
- Advanced refinement techniques
- Coarse-to-fine models (BiRefNetC2F)

## Example Use Cases

1. **Combined segmentation**: Foreground mask + edge map + detail map
2. **Multi-part segmentation**: Head + body + limbs for human figures
3. **Multiple instance segmentation**: Different objects in the same scene
4. **Feature-oriented segmentation**: Subject + shadow + reflection