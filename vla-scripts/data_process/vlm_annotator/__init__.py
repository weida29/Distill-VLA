"""
VLM Annotator - 三阶段自动标注模块

Stage 1: 正则提取物体名称 (text_parser)
Stage 2: Grounding DINO 检测 + 序号标注 (grounding_dino)
Stage 3: VLM 筛选判断 (qwen_vl_api)
"""

from .text_parser import extract_source_target, get_object_list
from .grounding_dino import GroundingDINODetector, Detection, get_box_by_label, draw_detections_on_image
from .qwen_vl_api import QwenVLAnnotator, GroundingResult, BoundingBox

__all__ = [
    # Stage 1
    'extract_source_target',
    'get_object_list',
    # Stage 2
    'GroundingDINODetector',
    'Detection',
    'get_box_by_label',
    # Stage 3 + Main
    'QwenVLAnnotator',
    'GroundingResult',
    'BoundingBox',
]
