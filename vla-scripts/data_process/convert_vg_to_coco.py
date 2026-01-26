#!/usr/bin/env python3
"""
Convert Visual Grounding (VG) JSONL format to COCO JSON format.
Required for Open-GroundingDino validation dataset.

VG format (input):
{
    "filename": "libero_object_no_noops/episode_xxx.jpg",
    "height": 256, "width": 256,
    "grounding": {
        "caption": "bowl. plate.",
        "regions": [{"phrase": "bowl", "bbox": [x1, y1, x2, y2]}]
    }
}

COCO format (output):
{
    "images": [{"id": 1, "file_name": "xxx.jpg", "height": H, "width": W}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [x, y, w, h], "area": A, "iscrowd": 0}],
    "categories": [{"id": 0, "name": "object"}]
}

Usage:
    python convert_vg_to_coco.py --input val.jsonl --output val_coco.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def convert_xyxy_to_xywh(bbox: List[int]) -> List[int]:
    """
    Convert bbox from [x1, y1, x2, y2] to [x, y, width, height].
    
    Args:
        bbox: [x1, y1, x2, y2] format
    
    Returns:
        [x, y, width, height] format
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def build_category_mapping(annotations: List[Dict]) -> Dict[str, int]:
    """
    Build a mapping from phrase to category_id.
    
    Args:
        annotations: List of VG annotations
    
    Returns:
        Dict mapping phrase -> category_id
    """
    phrases: Set[str] = set()
    
    for ann in annotations:
        if 'grounding' in ann:
            for region in ann['grounding'].get('regions', []):
                phrase = region.get('phrase', '').lower().strip()
                if phrase:
                    phrases.add(phrase)
    
    # Sort for consistency
    sorted_phrases = sorted(phrases)
    
    return {phrase: idx for idx, phrase in enumerate(sorted_phrases)}


def convert_vg_to_coco(
    input_path: str,
    output_path: str,
    category_output_path: str = None
) -> Dict:
    """
    Convert VG JSONL to COCO JSON format.
    
    Args:
        input_path: Path to input VG JSONL file
        output_path: Path to output COCO JSON file
        category_output_path: Optional path to save category mapping
    
    Returns:
        COCO format dict
    """
    # Load VG annotations
    annotations = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    
    print(f"Loaded {len(annotations)} annotations from {input_path}")
    
    # Build category mapping
    category_mapping = build_category_mapping(annotations)
    print(f"Found {len(category_mapping)} unique categories")
    
    # Build COCO structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for phrase, cat_id in category_mapping.items():
        coco["categories"].append({
            "id": cat_id,
            "name": phrase,
            "supercategory": "object"
        })
    
    # Convert images and annotations
    ann_id = 0
    
    for img_id, vg_ann in enumerate(annotations):
        filename = vg_ann['filename']
        height = vg_ann['height']
        width = vg_ann['width']
        
        # Add image
        coco["images"].append({
            "id": img_id,
            "file_name": filename,
            "height": height,
            "width": width
        })
        
        # Add annotations for this image
        if 'grounding' in vg_ann:
            for region in vg_ann['grounding'].get('regions', []):
                phrase = region.get('phrase', '').lower().strip()
                bbox_xyxy = region.get('bbox', [0, 0, 0, 0])
                
                if phrase not in category_mapping:
                    continue
                
                # Convert bbox to COCO format [x, y, w, h]
                bbox_xywh = convert_xyxy_to_xywh(bbox_xyxy)
                area = bbox_xywh[2] * bbox_xywh[3]
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_mapping[phrase],
                    "bbox": bbox_xywh,
                    "area": area,
                    "iscrowd": 0
                })
                ann_id += 1
    
    print(f"Created {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    
    # Save COCO JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    
    print(f"Saved COCO format to {output_path}")
    
    # Optionally save category mapping
    if category_output_path:
        # Format: {"0": "bowl", "1": "plate", ...}
        label_map = {str(v): k for k, v in category_mapping.items()}
        with open(category_output_path, 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)
        print(f"Saved label map to {category_output_path}")
    
    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert VG JSONL format to COCO JSON format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input VG JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output COCO JSON file path'
    )
    parser.add_argument(
        '--label_map',
        type=str,
        default=None,
        help='Optional: Output label map JSON file path'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Converting VG JSONL to COCO JSON Format")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.label_map:
        print(f"Label map: {args.label_map}")
    print("=" * 60)
    
    convert_vg_to_coco(args.input, args.output, args.label_map)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

