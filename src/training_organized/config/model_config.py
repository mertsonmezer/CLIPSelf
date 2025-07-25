"""
Model configuration for CLIPSelf training.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from training_organized.config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
  """Configuration for model settings."""

  # Core model settings
  model: str = "ViT-B-16"
  pretrained: str = "openai"
  precision: str = "amp"  # amp, fp16, fp32

  # Image processing
  force_image_size: Optional[Union[int, Tuple[int, int]]] = None
  image_mean: Optional[Tuple[float, float, float]] = None
  image_std: Optional[Tuple[float, float, float]] = None
  det_image_size: int = 1024

  # Model behavior
  force_quick_gelu: bool = False
  force_custom_text: bool = False
  force_patch_dropout: Optional[float] = None
  grad_checkpointing: bool = False

  # Locking settings
  lock_image: bool = False
  lock_image_unlocked_groups: int = 0
  lock_image_freeze_bn_stats: bool = False

  # Advanced settings
  torchscript: bool = False
  pretrained_image: bool = False
  cache_dir: Optional[str] = None

  def validate(self) -> None:
    """Validate model configuration."""
    super().validate()

    valid_precisions: List[str] = ["amp", "fp16", "fp32"]
    if self.precision not in valid_precisions:
      raise ValueError(f"precision must be one of {valid_precisions}, got {self.precision}")

    if self.force_patch_dropout is not None:
      if not (0.0 <= self.force_patch_dropout <= 1.0):
        raise ValueError("force_patch_dropout must be between 0.0 and 1.0")

    if self.lock_image_unlocked_groups < 0:
      raise ValueError("lock_image_unlocked_groups must be non-negative")
