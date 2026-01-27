"""
align_train/models/__init__.py

Models for VLA-GDINO alignment training.
"""

from .grounding_module import GroundingModule
from .gdino_teacher import GDINOTeacher

__all__ = ["GroundingModule", "GDINOTeacher"]

