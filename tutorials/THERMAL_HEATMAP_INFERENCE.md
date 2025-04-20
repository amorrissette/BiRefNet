# Thermal Heatmap Inference Tutorial

This tutorial explains how to run inference with a trained BiRefNet model on thermal images to generate heatmaps.

## Prerequisites

- A trained BiRefNet model checkpoint (`.pth` file)
- Thermal images (`.png` format)
- Optional: Thermal point data in JSON format (for ground truth comparison)

## Quick Start Demo

The easiest way to get started is to run the demo script:

```bash
python tutorials/demo_thermal_heatmap.py
```

This script will:
1. Find the latest model checkpoint in the `ckpt` directory
2. Find a sample thermal image in the test dataset
3. Run inference and save the results to the `output` directory

You can also specify your own checkpoint and image:

```bash
python tutorials/demo_thermal_heatmap.py --checkpoint path/to/model.pth --test_image path/to/thermal_image.png
```

## Direct Inference

For more control, you can use the inference script directly:

```bash
python tutorials/inference_thermal_heatmap.py --model path/to/model.pth --image path/to/thermal_image.png --json path/to/thermal_points.json --output results_dir
```

Arguments:
- `--model`: Path to the trained model weights (`.pth` file)
- `--image`: Path to the thermal image (`.png` file)
- `--json`: (Optional) Path to thermal points JSON file for ground truth comparison
- `--output`: Directory to save output results (default: `output`)

## Output Files

The inference script generates three types of output files:

1. **Prediction**: Raw prediction from the model (`pred_*.png`)
2. **Visualization**: Side-by-side comparison of original image, ground truth (if available), and prediction (`visualization_*.jpg`)
3. **Overlay**: Heatmap overlaid on the original image (`overlay_*.png`)

## JSON File Format

The JSON file format for thermal points should have the following structure:

```json
{
  "image_size": [640, 480],
  "circles": [
    {
      "center": [190, 78],
      "radius": 10.8,
      "intensity": 1.0  // Optional, defaults to 1.0
    },
    // More circles...
  ]
}
```

Alternatively:

```json
{
  "width": 640,
  "height": 480,
  "points": [
    {
      "x": 190, 
      "y": 78,
      "radius": 10.8,
      "intensity": 0.8  // Optional, defaults to 1.0
    },
    // More points...
  ]
}
```

## Using the Generated Heatmaps

The predicted heatmaps can be used for various applications:
- Visualizing thermal hotspots in an image
- Analyzing thermal patterns
- Input for further image processing or analysis

## Customizing Inference

You can modify the `inference_thermal_heatmap.py` script to change:
- The visualization colormap (currently uses 'hot' for predictions)
- The overlay transparency (currently alpha = 0.6)
- The output file formats and names