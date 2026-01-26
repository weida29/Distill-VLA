"""
ODVG (Object Detection Visual Grounding) Dataset for Grounding DINO fine-tuning.

Expected JSONL format:
{
    "filename": "images/xxx.jpg",
    "height": 256,
    "width": 256,
    "grounding": {
        "caption": "bowl. drawer.",
        "regions": [
            {"phrase": "bowl", "bbox": [0.1, 0.2, 0.3, 0.4]},
            {"phrase": "drawer", "bbox": [0.5, 0.6, 0.7, 0.8]}
        ]
    }
}
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F


class ODVGDataset(Dataset):
    """
    ODVG format dataset for Grounding DINO fine-tuning.
    
    Args:
        jsonl_path: Path to the JSONL annotation file
        image_dir: Root directory containing images
        transforms: Image transforms to apply
        max_text_len: Maximum text length for tokenization
    """
    
    def __init__(
        self,
        jsonl_path: str,
        image_dir: Optional[str] = None,
        transforms: Optional[Any] = None,
        max_text_len: int = 256,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.image_dir = Path(image_dir) if image_dir else self.jsonl_path.parent
        self.transforms = transforms
        self.max_text_len = max_text_len
        
        # Load annotations
        self.annotations = self._load_annotations()
        print(f"Loaded {len(self.annotations)} samples from {jsonl_path}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load JSONL annotations."""
        annotations = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(json.loads(line))
        return annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - image: Tensor of shape [3, H, W]
                - target: dict containing:
                    - boxes: Tensor of shape [N, 4] in cxcywh format, normalized 0-1
                    - labels: Tensor of shape [N] (all zeros for grounding)
                    - caption: str
                    - tokens_positive: List[List[int]] for each box
        """
        ann = self.annotations[idx]
        
        # Load image
        img_path = self.image_dir / ann['filename']
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Parse grounding info
        grounding = ann['grounding']
        caption = grounding['caption']
        regions = grounding['regions']
        
        # Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format
        # Boxes are already normalized to 0-1
        boxes = []
        phrases = []
        for region in regions:
            bbox = region['bbox']  # [x1, y1, x2, y2] normalized
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            boxes.append([cx, cy, bw, bh])
            phrases.append(region['phrase'])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Create tokens_positive: find the position of each phrase in the caption
        tokens_positive = self._get_tokens_positive(caption, phrases)
        
        # Apply transforms
        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)
        else:
            image = T.ToTensor()(image)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        target = {
            'boxes': boxes,
            'labels': torch.zeros(len(boxes), dtype=torch.long),  # All zeros for grounding
            'caption': caption,
            'tokens_positive': tokens_positive,
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor(image.shape[-2:]),
        }
        
        return image, target
    
    def _get_tokens_positive(self, caption: str, phrases: List[str]) -> List[List[int]]:
        """
        Find character-level positions of each phrase in the caption.
        Returns list of [start, end] positions for each phrase.
        """
        tokens_positive = []
        caption_lower = caption.lower()
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            start = caption_lower.find(phrase_lower)
            if start != -1:
                end = start + len(phrase)
                tokens_positive.append([start, end])
            else:
                # Phrase not found, use [0, 0] as placeholder
                tokens_positive.append([0, 0])
        
        return tokens_positive


class GroundingDINOTransform:
    """
    Transforms for Grounding DINO training.
    Applies random augmentations while keeping bbox coordinates consistent.
    """
    
    def __init__(
        self,
        image_size: int = 800,
        max_size: int = 1333,
        random_resize: bool = True,
        random_crop: bool = True,
        normalize: bool = True,
    ):
        self.image_size = image_size
        self.max_size = max_size
        self.random_resize = random_resize
        self.random_crop = random_crop
        self.normalize = normalize
        
        # Define resize scales
        self.scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    
    def __call__(self, image: Image.Image, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: PIL Image
            boxes: Tensor of shape [N, 4] in cxcywh format, normalized 0-1
        
        Returns:
            image: Tensor of shape [3, H, W]
            boxes: Tensor of shape [N, 4] in cxcywh format, normalized 0-1
        """
        w, h = image.size
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            # Flip boxes: cx -> 1 - cx
            boxes[:, 0] = 1 - boxes[:, 0]
        
        # Random resize
        if self.random_resize:
            target_size = random.choice(self.scales)
        else:
            target_size = self.image_size
        
        # Resize image
        image = self._resize(image, target_size, self.max_size)
        
        # Convert to tensor
        image = T.ToTensor()(image)
        
        # Normalize
        if self.normalize:
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        return image, boxes
    
    def _resize(self, image: Image.Image, target_size: int, max_size: int) -> Image.Image:
        """Resize image keeping aspect ratio."""
        w, h = image.size
        
        # Compute new size
        if w < h:
            new_w = target_size
            new_h = int(target_size * h / w)
        else:
            new_h = target_size
            new_w = int(target_size * w / h)
        
        # Clip to max size
        if new_w > max_size:
            new_w = max_size
            new_h = int(max_size * h / w)
        if new_h > max_size:
            new_h = max_size
            new_w = int(max_size * w / h)
        
        return image.resize((new_w, new_h), Image.BILINEAR)


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Custom collate function for variable-size images and targets.
    
    Args:
        batch: List of (image, target) tuples
    
    Returns:
        images: NestedTensor or padded Tensor
        targets: List of target dicts
    """
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    # Find max size
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad images
    batch_images = []
    masks = []
    for img in images:
        c, h, w = img.shape
        padded = torch.zeros(c, max_h, max_w, dtype=img.dtype)
        padded[:, :h, :w] = img
        batch_images.append(padded)
        
        # Create mask (True for padded regions)
        mask = torch.ones(max_h, max_w, dtype=torch.bool)
        mask[:h, :w] = False
        masks.append(mask)
    
    batch_images = torch.stack(batch_images)
    masks = torch.stack(masks)
    
    # Return as NestedTensor-like dict
    from groundingdino.util.misc import NestedTensor
    nested = NestedTensor(batch_images, masks)
    
    return nested, targets


def build_odvg_dataset(
    jsonl_path: str,
    image_dir: Optional[str] = None,
    image_size: int = 800,
    is_train: bool = True,
) -> ODVGDataset:
    """
    Build ODVG dataset with appropriate transforms.
    
    Args:
        jsonl_path: Path to JSONL file
        image_dir: Directory containing images
        image_size: Target image size
        is_train: Whether this is training set
    
    Returns:
        ODVGDataset instance
    """
    if is_train:
        transforms = GroundingDINOTransform(
            image_size=image_size,
            random_resize=True,
            random_crop=False,
            normalize=True,
        )
    else:
        transforms = GroundingDINOTransform(
            image_size=image_size,
            random_resize=False,
            random_crop=False,
            normalize=True,
        )
    
    return ODVGDataset(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        transforms=transforms,
    )


if __name__ == "__main__":
    # Test dataset loading
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "GroundingDINO"))
    
    # Test with sample data
    dataset_path = Path(__file__).parent.parent.parent / "data_processed" / "grounding_dino_dataset"
    train_jsonl = dataset_path / "train.jsonl"
    
    if train_jsonl.exists():
        dataset = build_odvg_dataset(
            jsonl_path=str(train_jsonl),
            image_dir=str(dataset_path),
            is_train=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Caption: {target['caption']}")
        print(f"Tokens positive: {target['tokens_positive']}")
    else:
        print(f"Dataset not found at {train_jsonl}")







