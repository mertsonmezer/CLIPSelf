"""
CLIPSelf-specific configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from training_organized.config import BaseConfig


@dataclass
class CLIPSelfConfig(BaseConfig):
  """Configuration specific to CLIPSelf method."""

  # CLIPSelf core settings
  extract_type: str = "v2"  # v1 and v2 for extraction type, v2 is default

  # Student-teacher ensemble
  alpha: float = 2.0  # >= 1.0 means no ensemble, < 1.0 enables ensemble

  multiscale: bool = False
  cosine_weight: float = 1.0  # Weight for cosine similarity loss

  # Multiscale training targets (when multiscale=True)
  # These will be automatically determined based on input size
  multiscale_targets_1024: Optional[List[int]] = None  # [320, 640, 896, 1024]
  multiscale_targets_896: Optional[List[int]] = None  # [336, 448, 672, 896]

  def __post_init__(self):
    """Initialize default multiscale targets if not provided."""
    if self.multiscale_targets_1024 is None:
      self.multiscale_targets_1024 = [320, 640, 896, 1024]
    if self.multiscale_targets_896 is None:
      self.multiscale_targets_896 = [336, 448, 672, 896]

  def validate(self) -> None:
    """Validate CLIPSelf configuration."""
    super().validate()

    valid_extract_types: List[str] = ["v1", "v2"]  # v1 and v2 supported
    if self.extract_type not in valid_extract_types:
      raise ValueError(f"extract_type must be one of {valid_extract_types}")

    if self.alpha < 0:
      raise ValueError("alpha must be non-negative")

    # Validate multiscale targets
    for targets, name in [
      (self.multiscale_targets_1024, "multiscale_targets_1024"),
      (self.multiscale_targets_896, "multiscale_targets_896"),
    ]:
      if targets is not None:
        if len(targets) == 0:
          raise ValueError(f"{name} must be a non-empty list")
        if not all(t > 0 for t in targets):
          raise ValueError(f"{name} must contain positive integers")
