"""
Loss configuration for CLIPSelf training.
"""

from dataclasses import dataclass

from training_organized.config import BaseConfig


@dataclass
class LossConfig(BaseConfig):
  """Configuration for loss functions and weights."""

  # Loss weights
  kl_weight: float = 1.0
  contrast_weight: float = 1.0
  l1_weight: float = 0.10
  smooth_weight: float = 0.0

  def validate(self) -> None:
    """Validate loss configuration."""
    super().validate()

    # All weights should be non-negative
    weights = [
      ("kl_weight", self.kl_weight),
      ("contrast_weight", self.contrast_weight),
      ("l1_weight", self.l1_weight),
      ("smooth_weight", self.smooth_weight),
    ]

    for name, weight in weights:
      if weight < 0:
        raise ValueError(f"{name} must be non-negative, got {weight}")
