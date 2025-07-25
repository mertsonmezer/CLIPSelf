"""
Base configuration class for CLIPSelf training.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class BaseConfig:
  """Base configuration class with common functionality."""

  def to_dict(self) -> Dict[str, Any]:
    """Convert config to dictionary."""
    return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

  def update_from_args(self, args: argparse.Namespace) -> None:
    """Update config from command line arguments."""
    for key, value in vars(args).items():
      if hasattr(self, key):
        setattr(self, key, value)

  def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
    """Update config from dictionary."""
    for key, value in config_dict.items():
      if hasattr(self, key):
        setattr(self, key, value)

  def validate(self) -> None:
    """Validate configuration. Override in subclasses."""
    pass
