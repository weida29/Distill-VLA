"""
align_train/models/__init__.py

Models for VLA-GDINO alignment training.
"""

from .grounding_module import GroundingModule
from .gdino_teacher import GDINOTeacher
from .action_query_alignment_head import ActionQueryAlignmentHead

__all__ = ["GroundingModule", "GDINOTeacher", "ActionQueryAlignmentHead"]



