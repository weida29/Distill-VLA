#!/usr/bin/env python3
"""
Convert bbox.json to Grounding DINO ODVG format dataset.

ODVG (Object Detection Visual Grounding) format:
{
  "filename": "images/episode_00016_frame_00000.jpg",
  "height": 480,
  "width": 640,
  "grounding": {
    "caption": "put the black bowl in the bottom drawer...",
    "regions": [
      {"phrase": "bowl", "bbox": [0.312, 0.594, 0.389, 0.714]},
      {"phrase": "drawer", "bbox": [0.123, 0.500, 0.248, 0.714]}
    ]
  }
}
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm


def load_bbox_jsonl(json_path: Path) -> List[Dict]:
    """Load bbox.json (JSONL format) data."""
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_bbox_to_normalized(
    bbox: List[int], 
    coord_scale: int = 1000
) -> List[float]:
    """
    Convert bbox from 0-1000 scale to 0-1 normalized coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] in 0-1000 scale
        coord_scale: Original coordinate scale (default 1000)
    
    Returns:
        [x1, y1, x2, y2] normalized to 0-1
    """
    return [coord / coord_scale for coord in bbox]


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """Get image width and height."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_to_odvg(
    item: Dict,
    source_image_path: Path,
    output_image_name: str,
    coord_scale: int = 1000
) -> Optional[Dict]:
    """
    Convert a single bbox item to ODVG format.
    
    Args:
        item: Original bbox data item
        source_image_path: Path to the source image
        output_image_name: Relative path for the output image
        coord_scale: Original coordinate scale
    
    Returns:
        ODVG format dict or None if conversion fails
    """
    if not source_image_path.exists():
        return None
    
    # Get image dimensions
    width, height = get_image_size(source_image_path)
    
    # Extract labels and bboxes (support both new and old formats)
    if 'bboxes_2d' in item['result'] and 'labels' in item['result']:
        bboxes = item['result']['bboxes_2d']
        labels = item['result']['labels']
    elif 'bbox_2d' in item['result'] and 'label' in item['result']:
        bboxes = [item['result']['bbox_2d']]
        labels = [item['result']['label']]
    else:
        return None
    
    # Ensure labels and bboxes have same length
    if len(labels) != len(bboxes):
        min_len = min(len(labels), len(bboxes))
        labels, bboxes = labels[:min_len], bboxes[:min_len]
    
    if not labels:
        return None
    
    # Build regions
    regions = []
    for label, bbox in zip(labels, bboxes):
        normalized_bbox = convert_bbox_to_normalized(bbox, coord_scale)
        regions.append({
            "phrase": label.strip(),
            "bbox": normalized_bbox
        })
    
    # Build ODVG format
    # Use labels as caption (e.g., "bowl. drawer.") for better grounding alignment
    caption = ". ".join([label.strip() for label in labels]) + "."
    
    odvg_item = {
        "filename": output_image_name,
        "height": height,
        "width": width,
        "grounding": {
            "caption": caption,
            "regions": regions
        }
    }
    
    # Also save original prompt if available (for evaluation)
    if 'prompt' in item:
        odvg_item["prompt"] = item['prompt']
    
    return odvg_item


def process_dataset(
    bbox_json_path: Path,
    keyframes_dir: Path,
    output_dir: Path,
    frame_idx: int = 0,
    val_ratio: float = 0.0,
    copy_images: bool = True,
    coord_scale: int = 1000,
    server_prefix: str = "/tmp/VLA-Adapter",
    local_prefix: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """
    Process the entire dataset and convert to ODVG format.
    
    Args:
        bbox_json_path: Path to bbox.json
        keyframes_dir: Directory containing keyframe images
        output_dir: Output directory for the dataset
        frame_idx: Which frame to use (default 0)
        val_ratio: Ratio of validation set (0.0 means no split)
        copy_images: Whether to copy images (True) or create symlinks (False)
        coord_scale: Original coordinate scale
        server_prefix: Server path prefix to replace
        local_prefix: Local path prefix to use
        seed: Random seed for train/val split
    
    Returns:
        Statistics dict
    """
    # Setup
    random.seed(seed)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine local prefix
    if local_prefix is None:
        local_prefix = str(bbox_json_path.parent.parent)
    
    def convert_path(server_path: str) -> Path:
        """Convert server path to local path."""
        local_path = server_path.replace(server_prefix, local_prefix)
        return Path(local_path)
    
    # Load data
    print(f"Loading bbox data from: {bbox_json_path}")
    bbox_data = load_bbox_jsonl(bbox_json_path)
    print(f"Loaded {len(bbox_data)} records")
    
    # Process each item
    odvg_data = []
    stats = {
        "total": len(bbox_data),
        "success": 0,
        "failed": 0,
        "skipped_no_image": 0,
        "skipped_no_bbox": 0,
        "total_objects": 0,
        "unique_labels": set(),
        "images_processed": set()
    }
    
    print("Converting to ODVG format...")
    for item in tqdm(bbox_data, desc="Processing"):
        # Get source image path
        episode_dir = convert_path(item['video_path'])
        source_image = episode_dir / f"frame_{frame_idx:05d}.jpg"
        
        # Generate output image name - use subdirectory for each task
        # e.g., /tmp/.../keyframes/libero_spatial_no_noops/episode_00344
        #       -> images/libero_spatial_no_noops/episode_00344_frame_00000.jpg
        video_path = Path(item['video_path'])
        task_name = video_path.parent.name  # e.g., "libero_spatial_no_noops"
        episode_name = video_path.name      # e.g., "episode_00344"
        output_image_name = f"images/{task_name}/{episode_name}_frame_{frame_idx:05d}.jpg"
        
        # Convert to ODVG
        odvg_item = convert_to_odvg(
            item, 
            source_image, 
            output_image_name,
            coord_scale
        )
        
        if odvg_item is None:
            if not source_image.exists():
                stats["skipped_no_image"] += 1
            else:
                stats["skipped_no_bbox"] += 1
            stats["failed"] += 1
            continue
        
        odvg_data.append(odvg_item)
        stats["success"] += 1
        stats["total_objects"] += len(odvg_item["grounding"]["regions"])
        
        # Track unique labels
        for region in odvg_item["grounding"]["regions"]:
            stats["unique_labels"].add(region["phrase"])
        
        # Copy or link image (skip if already processed)
        if source_image not in stats["images_processed"]:
            dest_image = output_dir / output_image_name
            dest_image.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_images:
                if not dest_image.exists():
                    shutil.copy2(source_image, dest_image)
            else:
                if not dest_image.exists():
                    dest_image.symlink_to(source_image.resolve())
            
            stats["images_processed"].add(source_image)
    
    # Train/val split
    if val_ratio > 0:
        random.shuffle(odvg_data)
        split_idx = int(len(odvg_data) * (1 - val_ratio))
        train_data = odvg_data[:split_idx]
        val_data = odvg_data[split_idx:]
        
        # Save train set
        train_path = output_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Save val set
        val_path = output_dir / "val.jsonl"
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        stats["train_count"] = len(train_data)
        stats["val_count"] = len(val_data)
        print(f"Saved train set: {train_path} ({len(train_data)} samples)")
        print(f"Saved val set: {val_path} ({len(val_data)} samples)")
    else:
        # Save all as train
        train_path = output_dir / "train.jsonl"
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in odvg_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        stats["train_count"] = len(odvg_data)
        stats["val_count"] = 0
        print(f"Saved dataset: {train_path} ({len(odvg_data)} samples)")
    
    # Convert set to list for JSON serialization
    stats["unique_labels"] = sorted(list(stats["unique_labels"]))
    stats["images_processed"] = len(stats["images_processed"])
    
    # Save metadata
    meta = {
        "format": "ODVG",
        "description": "Grounding DINO fine-tuning dataset",
        "coord_format": "normalized 0-1",
        "frame_idx": frame_idx,
        "statistics": {
            "total_samples": stats["success"],
            "train_samples": stats["train_count"],
            "val_samples": stats["val_count"],
            "total_objects": stats["total_objects"],
            "unique_labels": stats["unique_labels"],
            "num_unique_labels": len(stats["unique_labels"]),
            "images_copied": stats["images_processed"]
        }
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {meta_path}")
    
    return stats


def print_stats(stats: Dict):
    """Print processing statistics."""
    print("\n" + "=" * 50)
    print("Dataset Conversion Statistics")
    print("=" * 50)
    print(f"Total records:        {stats['total']}")
    print(f"Successfully converted: {stats['success']}")
    print(f"Failed:               {stats['failed']}")
    print(f"  - No image found:   {stats['skipped_no_image']}")
    print(f"  - No valid bbox:    {stats['skipped_no_bbox']}")
    print(f"Total objects:        {stats['total_objects']}")
    print(f"Unique labels:        {len(stats['unique_labels'])}")
    print(f"Images processed:     {stats['images_processed']}")
    if stats.get('val_count', 0) > 0:
        print(f"Train samples:        {stats['train_count']}")
        print(f"Val samples:          {stats['val_count']}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Convert bbox.json to Grounding DINO ODVG format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_to_grounding_dino.py
  python convert_to_grounding_dino.py --val-ratio 0.1
  python convert_to_grounding_dino.py --bbox-json path/to/bbox.json --output path/to/output
  python convert_to_grounding_dino.py --no-copy  # Use symlinks instead of copying
        """
    )
    
    parser.add_argument(
        '--bbox-json',
        type=str,
        default=None,
        help='Path to bbox.json file (default: data_processed/bbox.json)'
    )
    parser.add_argument(
        '--keyframes-dir',
        type=str,
        default=None,
        help='Directory containing keyframe images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data_processed/grounding_dino_dataset)'
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=0,
        help='Frame index to use (default: 0)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.0,
        help='Validation set ratio (default: 0.0, no split)'
    )
    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Create symlinks instead of copying images'
    )
    parser.add_argument(
        '--coord-scale',
        type=int,
        default=1000,
        help='Original coordinate scale (default: 1000)'
    )
    parser.add_argument(
        '--server-prefix',
        type=str,
        default='/tmp/VLA-Adapter',
        help='Server path prefix to replace'
    )
    parser.add_argument(
        '--local-prefix',
        type=str,
        default=None,
        help='Local path prefix (auto-detected if not provided)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Set default paths
    if args.bbox_json is None:
        bbox_json_path = project_root / "data_processed" / "bbox.json"
    else:
        bbox_json_path = Path(args.bbox_json)
    
    if args.keyframes_dir is None:
        keyframes_dir = project_root / "data_processed" / "keyframes"
    else:
        keyframes_dir = Path(args.keyframes_dir)
    
    if args.output is None:
        output_dir = project_root / "data_processed" / "grounding_dino_dataset"
    else:
        output_dir = Path(args.output)
    
    local_prefix = args.local_prefix or str(project_root)
    
    # Validate paths
    if not bbox_json_path.exists():
        print(f"Error: bbox.json not found at {bbox_json_path}")
        return 1
    
    print(f"Project root: {project_root}")
    print(f"Bbox JSON: {bbox_json_path}")
    print(f"Output dir: {output_dir}")
    print(f"Frame index: {args.frame_idx}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Copy images: {not args.no_copy}")
    print()
    
    # Process dataset
    stats = process_dataset(
        bbox_json_path=bbox_json_path,
        keyframes_dir=keyframes_dir,
        output_dir=output_dir,
        frame_idx=args.frame_idx,
        val_ratio=args.val_ratio,
        copy_images=not args.no_copy,
        coord_scale=args.coord_scale,
        server_prefix=args.server_prefix,
        local_prefix=local_prefix,
        seed=args.seed
    )
    
    print_stats(stats)
    
    print(f"\nDataset saved to: {output_dir}")
    print("Done!")
    
    return 0


if __name__ == "__main__":
    exit(main())

