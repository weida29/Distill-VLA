#!/usr/bin/env python3
"""
Download Grounding DINO pretrained weights.

Usage:
    python download_pretrained.py
    python download_pretrained.py --model swinb
"""

import argparse
import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Pretrained model URLs
PRETRAINED_URLS = {
    "swint": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "filename": "groundingdino_swint_ogc.pth",
        "config": "GroundingDINO_SwinT_OGC.py",
        "description": "Grounding DINO with Swin-T backbone (lightweight)",
    },
    "swinb": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
        "filename": "groundingdino_swinb_cogcoor.pth",
        "config": "GroundingDINO_SwinB_cfg.py",
        "description": "Grounding DINO with Swin-B backbone (larger, more accurate)",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Grounding DINO pretrained weights")
    parser.add_argument(
        "--model",
        type=str,
        choices=["swint", "swinb", "all"],
        default="swint",
        help="Model version to download (default: swint)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: visual_teacher/pretrained_ckpt)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "pretrained_ckpt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine which models to download
    if args.model == "all":
        models_to_download = list(PRETRAINED_URLS.keys())
    else:
        models_to_download = [args.model]
    
    # Download each model
    for model_name in models_to_download:
        model_info = PRETRAINED_URLS[model_name]
        output_path = output_dir / model_info["filename"]
        
        print(f"\n{'=' * 50}")
        print(f"Model: {model_name}")
        print(f"Description: {model_info['description']}")
        print(f"Config: {model_info['config']}")
        print(f"{'=' * 50}")
        
        if output_path.exists() and not args.force:
            print(f"File already exists: {output_path}")
            print("Use --force to re-download")
            continue
        
        print(f"Downloading from: {model_info['url']}")
        print(f"Saving to: {output_path}")
        
        success = download_file(model_info["url"], output_path)
        
        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"Download complete! File size: {file_size:.1f} MB")
        else:
            print(f"Download failed!")
    
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    for model_name in models_to_download:
        model_info = PRETRAINED_URLS[model_name]
        output_path = output_dir / model_info["filename"]
        status = "✓" if output_path.exists() else "✗"
        print(f"{status} {model_name}: {output_path}")
    
    print("\nUsage:")
    print("  python train_grounding_dino.py --pretrained ../pretrained_ckpt/groundingdino_swint_ogc.pth")


if __name__ == "__main__":
    main()







