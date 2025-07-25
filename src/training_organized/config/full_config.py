"""
Comprehensive configuration management for CLIPSelf training.
"""

import argparse
import logging
from dataclasses import dataclass

from training_organized.config import BaseConfig, CLIPSelfConfig, DataConfig, LossConfig, ModelConfig, TrainingConfig


@dataclass
class CLIPSelfFullConfig(BaseConfig):
  """Complete configuration for CLIPSelf training."""

  # Sub-configurations
  model: ModelConfig
  training: TrainingConfig
  data: DataConfig
  clipself: CLIPSelfConfig
  loss: LossConfig

  # Global settings
  device: str = "cuda"

  def __init__(self):
    """Initialize with default sub-configurations."""
    self.model = ModelConfig()
    self.training = TrainingConfig()
    self.data = DataConfig()
    self.clipself = CLIPSelfConfig()
    self.loss = LossConfig()

  def validate(self) -> None:
    """Validate all sub-configurations."""
    super().validate()
    self.model.validate()
    self.training.validate()
    self.data.validate()
    self.clipself.validate()
    self.loss.validate()

    # Cross-validation between configs
    if self.data.dataset_type not in ["proposals_distill", "grid_distill"]:
      raise ValueError("Only proposals_distill and grid_distill are supported for CLIPSelf")

  def update_from_args(self, args: argparse.Namespace) -> None:
    """Update configuration from command line arguments."""
    # Update each sub-config
    self.model.update_from_args(args)
    self.training.update_from_args(args)
    self.data.update_from_args(args)
    self.clipself.update_from_args(args)
    self.loss.update_from_args(args)

    # Update global settings
    if hasattr(args, "device"):
      self.device = args.device

  def log_config(self) -> None:
    """Log the current configuration."""
    logging.info("=== CLIPSelf Configuration ===")
    logging.info("Model Config:")
    for key, value in self.model.to_dict().items():
      logging.info(f"  {key}: {value}")

    logging.info("Training Config:")
    for key, value in self.training.to_dict().items():
      logging.info(f"  {key}: {value}")

    logging.info("Data Config:")
    for key, value in self.data.to_dict().items():
      logging.info(f"  {key}: {value}")

    logging.info("CLIPSelf Config:")
    for key, value in self.clipself.to_dict().items():
      logging.info(f"  {key}: {value}")

    logging.info("Loss Config:")
    for key, value in self.loss.to_dict().items():
      logging.info(f"  {key}: {value}")

    logging.info(f"Device: {self.device}")
    logging.info("=" * 30)


def create_clipself_config_from_args(args: argparse.Namespace) -> CLIPSelfFullConfig:
  """Create and validate a CLIPSelf configuration from command line arguments."""
  config = CLIPSelfFullConfig()
  config.update_from_args(args)
  config.validate()
  return config
