"""Pipelines de segmentación para vision_engine."""

from .yolo_pipeline import (
    YoloArmDetector,
    YoloHumanParseDetector,
    FastSamArmDetector,
    YoloPipelineConfig,
)
from .rvm_background import (
    RvmConfig,
    RvmProcessor,
    BackgroundSubtractionConfig,
    BackgroundSubtractionProcessor,
)

__all__ = [
    "YoloArmDetector",
    "YoloHumanParseDetector",
    "FastSamArmDetector",
    "YoloPipelineConfig",
    "RvmConfig",
    "RvmProcessor",
    "BackgroundSubtractionConfig",
    "BackgroundSubtractionProcessor",
]
