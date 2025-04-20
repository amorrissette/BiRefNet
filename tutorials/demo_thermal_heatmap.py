#!/usr/bin/env python
# Demo script to demonstrate how to use the thermal heatmap inference

import os
import sys
import argparse
import glob

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutorials.inference_thermal_heatmap import run_inference

def main():
    parser = argparse.ArgumentParser(description='Demo script for thermal heatmap inference')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint (.pth file). If not provided, will use the latest one in ckpt folder')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to test thermal image. If not provided, will use sample image from test_dataset')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Find model checkpoint if not specified
    if args.checkpoint is None:
        print("Looking for latest checkpoint...")
        checkpoint_folders = sorted(glob.glob(os.path.join('ckpt', '*')))
        if not checkpoint_folders:
            print("No checkpoint folders found in 'ckpt' directory. Please specify a checkpoint with --checkpoint.")
            return
        
        latest_folder = checkpoint_folders[-1]
        checkpoint_files = sorted(glob.glob(os.path.join(latest_folder, '*.pth')))
        if not checkpoint_files:
            print(f"No .pth files found in {latest_folder}. Please specify a checkpoint with --checkpoint.")
            return
        
        checkpoint = checkpoint_files[-1]
        print(f"Using checkpoint: {checkpoint}")
    else:
        checkpoint = args.checkpoint
    
    # Find test image if not specified
    if args.test_image is None:
        print("Looking for sample test image...")
        # Try to find a sample thermal image in the test_dataset
        sample_images = glob.glob('test_dataset/Thermal/Thermal-TR/im/*_thermal.png')
        if not sample_images:
            print("No sample thermal images found in test_dataset. Please specify an image with --test_image.")
            return
        
        test_image = sample_images[0]
        print(f"Using test image: {test_image}")
    else:
        test_image = args.test_image
    
    # Find corresponding JSON file if it exists
    json_file = test_image.replace('/im/', '/gt/').replace('_thermal.png', '_thermal-tsc.json')
    if not os.path.exists(json_file):
        # Try the repository root
        base_name = os.path.basename(test_image).replace('_thermal.png', '_thermal-tsc.json')
        if os.path.exists(base_name):
            json_file = base_name
        else:
            print(f"No corresponding JSON file found for {test_image}. Will run without ground truth visualization.")
            json_file = None
    
    # Run inference
    print("\nRunning inference...")
    output_paths = run_inference(checkpoint, test_image, json_file, args.output)
    
    print("\nInference completed successfully!")
    print("Results saved to:")
    for path in output_paths:
        print(f"  - {path}")
    
    print("\nTo run inference on your own thermal images:")
    print(f"  python tutorials/inference_thermal_heatmap.py --model {checkpoint} --image path/to/your/thermal_image.png [--json path/to/thermal_points.json]")

if __name__ == "__main__":
    main()