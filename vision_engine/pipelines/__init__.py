"""Pipelines de segmentación para vision_engine."""

from vision_engine.pipelines.yolo_pipeline import YoloArmDetector, FastSamArmDetector, YoloPipelineConfig
from vision_engine.pipelines.rvm_background import (
    RvmConfig,
    RvmProcessor,
    BackgroundSubtractionConfig,
    BackgroundSubtractionProcessor,
)

__all__ = [
    "YoloArmDetector",
    "FastSamArmDetector",
    "YoloPipelineConfig",
    "RvmConfig",
    "RvmProcessor",
    "BackgroundSubtractionConfig",
    "BackgroundSubtractionProcessor",
]
