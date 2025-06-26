# pytracking/soi_pipeline/__init__.py
"""
SOI Pipeline for PyTracking Framework
Single Object Tracking with Semantic Object Identification
"""

__version__ = "1.0.0"
__author__ = "SOI Research Team"

from .core.box_utils import BoundingBox, compute_iou
from .core.data_processor import DataProcessor
from .core.frame_extractor import FrameExtractor
from .models.vlm_interface import VLMEngine
from .configs.config import Config

__all__ = [
    'BoundingBox', 'compute_iou', 'DataProcessor', 'FrameExtractor', 
    'VLMEngine', 'Config'
]

