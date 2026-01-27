#!/usr/bin/env python3
"""
Convert grounding DINO dataset from normalized bbox format to pixel coordinate format
for Open-GroundingDino training.

Input format (normalized 0-1):
{
    "filename": "images/xxx.jpg",
    "height": 256, "width": 256,
    "grounding": {
        "caption": "bowl. plate.",
        "regions": [{"phrase": "bowl", "bbox": [0.406, 0.375, 0.462, 0.481]}]
    },
    "prompt": "pick up the bowl and place it on the plate"  # optional
}

Output format (pixel coordinates):
{
    "filename": "xxx.jpg",
    "height": 256, "width": 256,
    "grounding": {
        "caption": "pick up the bowl and place it on the plate",  # uses prompt if available
        "regions": [{"phrase": "bowl", "bbox": [104, 96, 118, 123]}]  # phrase is the object
    }
}

Note: The caption field will use the original prompt (instruction) if available,
      otherwise it falls back to the original caption. The phrase in regions
      remains the object name.

Usage:
    python convert_to_open_gdino.py --input_dir data_processed/grounding_dino_dataset \
                                    --output_dir data_processed/open_gdino_dataset \
                                    --visualize --num_vis 10
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def convert_bbox_normalized_to_pixel(
    bbox: List[float], 
    width: int, 
    height: int
) -> List[int]:
    """
    Convert bbox from normalized [0,1] format to pixel coordinates.
    
    Args:
        bbox: [x1_norm, y1_norm, x2_norm, y2_norm] in range [0, 1]
        width: image width in pixels
        height: image height in pixels
    
    Returns:
        [x1_px, y1_px, x2_px, y2_px] in pixel coordinates
    """
    x1_norm, y1_norm, x2_norm, y2_norm = bbox
    
    x1_px = int(round(x1_norm * width))
    y1_px = int(round(y1_norm * height))
    x2_px = int(round(x2_norm * width))
    y2_px = int(round(y2_norm * height))
    
    # Ensure valid bbox (x2 > x1, y2 > y1)
    x1_px = max(0, min(x1_px, width - 1))
    y1_px = max(0, min(y1_px, height - 1))
    x2_px = max(x1_px + 1, min(x2_px, width))
    y2_px = max(y1_px + 1, min(y2_px, height))
    
    return [x1_px, y1_px, x2_px, y2_px]


def convert_annotation(ann: Dict, image_root_prefix: str = "") -> Dict:
    """
    Convert a single annotation from normalized to pixel format.
    
    Args:
        ann: Original annotation dict
        image_root_prefix: Prefix to remove from filename path
    
    Returns:
        Converted annotation dict
    """
    width = ann['width']
    height = ann['height']
    
    # Convert filename path
    filename = ann['filename']
    if image_root_prefix and filename.startswith(image_root_prefix):
        filename = filename[len(image_root_prefix):]
    # Remove leading slash if present
    filename = filename.lstrip('/')
    
    # Build new annotation
    new_ann = {
        'filename': filename,
        'height': height,
        'width': width,
    }
    
    # Convert grounding regions
    if 'grounding' in ann:
        grounding = ann['grounding']
        new_regions = []
        
        for region in grounding.get('regions', []):
            bbox_norm = region['bbox']
            bbox_pixel = convert_bbox_normalized_to_pixel(bbox_norm, width, height)
            
            new_regions.append({
                'phrase': region['phrase'],
                'bbox': bbox_pixel
            })
        
        # Use original prompt as caption if available, otherwise fall back to original caption
        # This ensures caption is the instruction/prompt, while phrase remains the object name
        caption = ann.get('prompt', grounding.get('caption', ''))
        
        new_ann['grounding'] = {
            'caption': caption,
            'regions': new_regions
        }
    
    return new_ann


def convert_jsonl_file(
    input_path: str, 
    output_path: str, 
    image_root_prefix: str = "images/"
) -> List[Dict]:
    """
    Convert a JSONL file from normalized to pixel format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        image_root_prefix: Prefix to remove from filename
    
    Returns:
        List of converted annotations
    """
    annotations = []
    
    print(f"Converting: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                ann = json.loads(line)
                converted = convert_annotation(ann, image_root_prefix)
                annotations.append(converted)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(annotations)} annotations -> {output_path}")
    
    return annotations


def visualize_annotations(
    annotations: List[Dict],
    image_dir: str,
    output_dir: str,
    num_samples: int = 10,
    seed: int = 42
) -> None:
    """
    Visualize converted annotations to verify correctness.
    
    Args:
        annotations: List of converted annotations
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        seed: Random seed for sampling
    """
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample annotations
    if len(annotations) > num_samples:
        samples = random.sample(annotations, num_samples)
    else:
        samples = annotations
    
    # Define colors for different phrases
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    print(f"\nVisualizing {len(samples)} samples...")
    
    for idx, ann in enumerate(samples):
        filename = ann['filename']
        image_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bboxes
        if 'grounding' in ann:
            caption = ann['grounding'].get('caption', '')
            regions = ann['grounding'].get('regions', [])
            
            for i, region in enumerate(regions):
                bbox = region['bbox']
                phrase = region['phrase']
                color = colors[i % len(colors)]
                
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{phrase}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                
                cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
            
            # Add caption at bottom
            caption_text = f"Caption: {caption}"
            cv2.putText(image, caption_text, (5, image.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Save visualization
        output_filename = f"sample_{idx:03d}_{Path(filename).stem}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        
        print(f"  Saved: {output_path}")
    
    print(f"\nVisualization complete! Check {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert grounding DINO dataset to Open-GroundingDino format"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data_processed/grounding_dino_dataset',
        help='Input directory containing train.jsonl and val.jsonl'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data_processed/open_gdino_dataset',
        help='Output directory for converted files'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data_processed/grounding_dino_dataset/images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--image_root_prefix',
        type=str,
        default='images/',
        help='Prefix to remove from filename paths'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization to verify conversion'
    )
    parser.add_argument(
        '--num_vis',
        type=int,
        default=10,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Converting Grounding DINO Dataset to Open-GroundingDino Format")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image directory:  {args.image_dir}")
    print("=" * 60)
    
    all_annotations = []
    
    # Convert train.jsonl
    train_input = input_dir / 'train.jsonl'
    if train_input.exists():
        train_output = output_dir / 'train.jsonl'
        train_anns = convert_jsonl_file(
            str(train_input), 
            str(train_output),
            args.image_root_prefix
        )
        all_annotations.extend(train_anns)
    else:
        print(f"Warning: {train_input} not found")
    
    # Convert val.jsonl
    val_input = input_dir / 'val.jsonl'
    if val_input.exists():
        val_output = output_dir / 'val.jsonl'
        val_anns = convert_jsonl_file(
            str(val_input), 
            str(val_output),
            args.image_root_prefix
        )
        all_annotations.extend(val_anns)
    else:
        print(f"Warning: {val_input} not found")
    
    print(f"\nTotal annotations converted: {len(all_annotations)}")
    
    # Visualize if requested
    if args.visualize and all_annotations:
        vis_dir = output_dir / 'visualization'
        visualize_annotations(
            all_annotations,
            args.image_dir,
            str(vis_dir),
            num_samples=args.num_vis,
            seed=args.seed
        )
    
    print("\nDone!")
    
    # Print sample output for verification
    if all_annotations:
        print("\n" + "=" * 60)
        print("Sample converted annotation:")
        print("=" * 60)
        print(json.dumps(all_annotations[0], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

