"""
Extract keyframe images and prompts from LIBERO RLDS dataset.
This script extracts images at fixed intervals (keyframes) and organizes them
by episode, with each episode folder containing its language instruction (prompt).

Output structure:
    output_dir/
    ├── libero_spatial_no_noops/
    │   ├── episode_00000/
    │   │   ├── prompt.txt
    │   │   ├── frame_00000.jpg
    │   │   ├── frame_00005.jpg
    │   │   └── ...
    │   └── episode_00001/
    │       └── ...
    ├── libero_goal_no_noops/
    │   └── ...
    └── metadata.json
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from PIL import Image


def extract_keyframes_from_subset(
    data_dir: str,
    output_dir: str,
    subset_name: str,
    sample_rate: int = 5,
    max_episodes: int = None,
    save_wrist_image: bool = False
) -> dict:
    """
    Extract keyframe images and prompts from a single subset.
    
    Args:
        data_dir: Path to the RLDS dataset directory
        output_dir: Path to save extracted data
        subset_name: Name of the subset (e.g., 'libero_spatial_no_noops')
        sample_rate: Sample every N frames (default: 5)
        max_episodes: Maximum number of episodes to process (None for all)
        save_wrist_image: Whether to also save wrist camera images
    
    Returns:
        Dictionary containing statistics for this subset
    """
    import tensorflow_datasets as tfds
    
    # Build dataset path - RLDS format stores data in version subdirectory (e.g., 1.0.0)
    dataset_path = Path(data_dir) / subset_name
    
    if not dataset_path.exists():
        print(f"  [Warning] Dataset path not found: {dataset_path}")
        return {
            'subset': subset_name,
            'episodes': 0,
            'total_frames': 0,
            'sampled_frames': 0,
            'error': f'Dataset path not found: {dataset_path}'
        }
    
    # Find version directory (e.g., 1.0.0) containing dataset_info.json
    version_dir = None
    for subdir in dataset_path.iterdir():
        if subdir.is_dir() and (subdir / 'dataset_info.json').exists():
            version_dir = subdir
            break
    
    if version_dir is None:
        print(f"  [Warning] No version directory with dataset_info.json found in: {dataset_path}")
        return {
            'subset': subset_name,
            'episodes': 0,
            'total_frames': 0,
            'sampled_frames': 0,
            'error': f'No dataset_info.json found in {dataset_path}'
        }
    
    print(f"  Found version directory: {version_dir.name}")
    
    # Setup output directory for this subset
    subset_output_dir = Path(output_dir) / subset_name
    subset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset using tensorflow_datasets - point to version directory
    try:
        builder = tfds.builder_from_directory(str(version_dir))
        ds = builder.as_dataset(split='train')
    except Exception as e:
        print(f"  [Error] Failed to load dataset: {e}")
        return {
            'subset': subset_name,
            'episodes': 0,
            'total_frames': 0,
            'sampled_frames': 0,
            'error': str(e)
        }
    
    # Statistics
    episode_count = 0
    total_frames = 0
    sampled_frames = 0
    episode_stats = []
    
    # Process each episode
    for episode in tqdm(ds, desc=f"  Processing {subset_name}"):
        if max_episodes and episode_count >= max_episodes:
            break
        
        # Create episode directory
        episode_dir = subset_output_dir / f"episode_{episode_count:05d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        steps = episode['steps']
        language_instruction = None
        frame_idx = 0
        episode_sampled_frames = 0
        
        for step in steps:
            total_frames += 1
            
            # Get language instruction (same for all steps in episode)
            if language_instruction is None:
                lang = step['language_instruction'].numpy()
                if isinstance(lang, bytes):
                    language_instruction = lang.decode('utf-8')
                else:
                    language_instruction = str(lang)
            
            # Sample frames at fixed intervals
            if frame_idx % sample_rate == 0:
                # Get main camera image
                image_data = step['observation']['image'].numpy()
                
                if image_data is not None and isinstance(image_data, np.ndarray):
                    # Save main image
                    image_filename = f"frame_{frame_idx:05d}.jpg"
                    image_path = episode_dir / image_filename
                    
                    img = Image.fromarray(image_data)
                    img.save(str(image_path), quality=95)
                    
                    sampled_frames += 1
                    episode_sampled_frames += 1
                    
                    # Optionally save wrist image
                    if save_wrist_image:
                        wrist_image_data = step['observation']['wrist_image'].numpy()
                        if wrist_image_data is not None and isinstance(wrist_image_data, np.ndarray):
                            wrist_filename = f"wrist_{frame_idx:05d}.jpg"
                            wrist_path = episode_dir / wrist_filename
                            wrist_img = Image.fromarray(wrist_image_data)
                            wrist_img.save(str(wrist_path), quality=95)
            
            frame_idx += 1
        
        # Save prompt (language instruction) for this episode
        prompt_file = episode_dir / "prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(language_instruction or "")
        
        # Record episode statistics
        episode_stats.append({
            'episode_id': episode_count,
            'language_instruction': language_instruction,
            'total_frames': frame_idx,
            'sampled_frames': episode_sampled_frames
        })
        
        episode_count += 1
    
    # Return subset statistics
    return {
        'subset': subset_name,
        'episodes': episode_count,
        'total_frames': total_frames,
        'sampled_frames': sampled_frames,
        'sample_rate': sample_rate,
        'episode_stats': episode_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract keyframe images and prompts from LIBERO RLDS dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract with default settings (sample every 5 frames)
  python extract_keyframes.py
  
  # Extract every 10 frames from specific subsets
  python extract_keyframes.py --sample_rate 10 --subsets libero_goal_no_noops libero_10_no_noops
  
  # Extract only first 10 episodes from each subset
  python extract_keyframes.py --max_episodes 10
  
  # Also save wrist camera images
  python extract_keyframes.py --save_wrist_image
        """
    )
    parser.add_argument(
        '--data_dir', type=str,
        default='data/modified_libero_rlds',
        help='Path to RLDS dataset directory (default: data/modified_libero_rlds)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='data_processed/keyframes',
        help='Output directory for extracted data (default: data_processed/keyframes)'
    )
    parser.add_argument(
        '--sample_rate', type=int, default=5,
        help='Sample every N frames (default: 5)'
    )
    parser.add_argument(
        '--max_episodes', type=int, default=None,
        help='Maximum episodes per subset (default: all)'
    )
    parser.add_argument(
        '--subsets', type=str, nargs='+',
        default=['libero_spatial_no_noops', 'libero_object_no_noops',
                 'libero_goal_no_noops', 'libero_10_no_noops'],
        help='Subsets to process'
    )
    parser.add_argument(
        '--save_wrist_image', action='store_true',
        help='Also save wrist camera images'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LIBERO Keyframe Extraction")
    print("=" * 60)
    print(f"Data directory:    {args.data_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Sample rate:       every {args.sample_rate} frames")
    print(f"Max episodes:      {args.max_episodes or 'all'}")
    print(f"Subsets:           {', '.join(args.subsets)}")
    print(f"Save wrist images: {args.save_wrist_image}")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all subsets
    all_stats = []
    total_episodes = 0
    total_sampled_frames = 0
    
    for subset in args.subsets:
        print(f"\n{'─' * 60}")
        print(f"Processing subset: {subset}")
        print(f"{'─' * 60}")
        
        stats = extract_keyframes_from_subset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            subset_name=subset,
            sample_rate=args.sample_rate,
            max_episodes=args.max_episodes,
            save_wrist_image=args.save_wrist_image
        )
        
        all_stats.append(stats)
        total_episodes += stats.get('episodes', 0)
        total_sampled_frames += stats.get('sampled_frames', 0)
        
        print(f"  Episodes:       {stats.get('episodes', 0)}")
        print(f"  Total frames:   {stats.get('total_frames', 0)}")
        print(f"  Sampled frames: {stats.get('sampled_frames', 0)}")
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'sample_rate': args.sample_rate,
        'max_episodes': args.max_episodes,
        'save_wrist_image': args.save_wrist_image,
        'subsets': all_stats,
        'summary': {
            'total_episodes': total_episodes,
            'total_sampled_frames': total_sampled_frames
        }
    }
    
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Total episodes:       {total_episodes}")
    print(f"Total sampled frames: {total_sampled_frames}")
    print(f"Metadata saved to:    {metadata_file}")
    print(f"Output directory:     {output_path.resolve()}")


if __name__ == '__main__':
    main()

