"""
Extract images and language instructions from LIBERO RLDS dataset.
This script reads tfrecord files and saves images as JPG files along with
corresponding text annotations.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image


def extract_from_rlds_dataset(data_dir, output_dir, subset_name, sample_rate=5, max_episodes=None):
    """
    Extract images and text from RLDS dataset.
    
    Args:
        data_dir: Path to the RLDS dataset directory
        output_dir: Path to save extracted images and annotations
        subset_name: Name of the subset (e.g., 'libero_spatial_no_noops')
        sample_rate: Sample every N frames to reduce redundancy
        max_episodes: Maximum number of episodes to process (None for all)
    """
    import tensorflow_datasets as tfds
    
    # Setup output directories
    images_dir = Path(output_dir) / "images" / subset_name
    images_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    image_id = 0
    
    # Build dataset path
    dataset_path = Path(data_dir) / subset_name
    
    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return []
    
    # Find version directory (e.g., 1.0.0) containing dataset_info.json
    version_dir = None
    for subdir in dataset_path.iterdir():
        if subdir.is_dir() and (subdir / 'dataset_info.json').exists():
            version_dir = subdir
            break
    
    if version_dir is None:
        print(f"  [Warning] No version directory with dataset_info.json found in: {dataset_path}")
        return []
    
    print(f"  Found version directory: {version_dir.name}")
    
    # Load dataset using tensorflow_datasets - point to version directory
    try:
        builder = tfds.builder_from_directory(str(version_dir))
        ds = builder.as_dataset(split='train')
    except Exception as e:
        print(f"  [Error] Failed to load dataset: {e}")
        return []
    
    episode_count = 0
    for episode in tqdm(ds, desc=f"Processing {subset_name}"):
        if max_episodes and episode_count >= max_episodes:
            break
            
        steps = episode['steps']
        language_instruction = None
        frame_idx = 0
        
        for step in steps:
            # Get language instruction (same for all steps in episode)
            if language_instruction is None:
                lang = step['language_instruction'].numpy()
                if isinstance(lang, bytes):
                    language_instruction = lang.decode('utf-8')
                else:
                    language_instruction = str(lang)
            
            # Sample frames
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue
            
            # Get main camera image
            image_data = step['observation']['image'].numpy()
            
            if image_data is not None and len(image_data) > 0:
                # Save image
                image_filename = f"{subset_name}_{episode_count:05d}_{frame_idx:05d}.jpg"
                image_path = images_dir / image_filename
                
                # Convert to PIL Image and save
                if isinstance(image_data, np.ndarray):
                    img = Image.fromarray(image_data)
                    img.save(str(image_path), quality=95)
                    
                    # Add annotation
                    annotations.append({
                        'image_id': image_id,
                        'file_name': str(Path(subset_name) / image_filename),
                        'width': image_data.shape[1],
                        'height': image_data.shape[0],
                        'language_instruction': language_instruction,
                        'episode_id': episode_count,
                        'frame_idx': frame_idx,
                        'subset': subset_name
                    })
                    
                    image_id += 1
            
            frame_idx += 1
        
        episode_count += 1
    
    print(f"Extracted {len(annotations)} images from {subset_name}")
    return annotations


def main():
    parser = argparse.ArgumentParser(description='Extract images from LIBERO RLDS dataset')
    parser.add_argument('--data_dir', type=str, 
                        default='data/modified_libero_rlds',
                        help='Path to RLDS dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='data_processed',
                        help='Output directory for extracted data')
    parser.add_argument('--sample_rate', type=int, default=5,
                        help='Sample every N frames (default: 5)')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum episodes per subset (default: all)')
    parser.add_argument('--subsets', type=str, nargs='+',
                        default=['libero_spatial_no_noops', 'libero_object_no_noops', 
                                'libero_goal_no_noops', 'libero_10_no_noops'],
                        help='Subsets to process')
    
    args = parser.parse_args()
    
    all_annotations = []
    
    for subset in args.subsets:
        print(f"\n{'='*50}")
        print(f"Processing subset: {subset}")
        print(f"{'='*50}")
        
        annotations = extract_from_rlds_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            subset_name=subset,
            sample_rate=args.sample_rate,
            max_episodes=args.max_episodes
        )
        all_annotations.extend(annotations)
    
    # Save all annotations
    annotations_file = Path(args.output_dir) / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Total images extracted: {len(all_annotations)}")
    print(f"Annotations saved to: {annotations_file}")


if __name__ == '__main__':
    main()
