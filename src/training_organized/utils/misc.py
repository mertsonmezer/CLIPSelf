"""
Miscellaneous utility functions.
"""

import random
import os
import logging
from typing import Union

import torch
import numpy as np


def set_random_seed(seed: int, rank: int = 0) -> None:
  """
  Set random seed for reproducibility.

  Args:
      seed: Base random seed
      rank: Process rank for distributed training
  """
  # Adjust seed by rank to ensure different seeds across processes
  effective_seed = seed + rank

  # Set seeds for all random number generators
  torch.manual_seed(effective_seed)
  np.random.seed(effective_seed)
  random.seed(effective_seed)

  # For CUDA
  if torch.cuda.is_available():
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

  logging.info(f"Random seed set to {effective_seed} (base={seed}, rank={rank})")


def get_device(device_str: str = "auto") -> torch.device:
  """
  Get the appropriate torch device.

  Args:
      device_str: Device specification ("auto", "cuda", "cpu", or specific device)

  Returns:
      torch.device object
  """
  if device_str == "auto":
    if torch.cuda.is_available():
      device = torch.device("cuda")
      logging.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
      device = torch.device("cpu")
      logging.info("CUDA not available, using CPU")
  else:
    device = torch.device(device_str)
    logging.info(f"Using specified device: {device}")

  return device


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
  """
  Count the number of parameters in a model.

  Args:
      model: PyTorch model
      trainable_only: Whether to count only trainable parameters

  Returns:
      Number of parameters
  """
  if trainable_only:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  else:
    return sum(p.numel() for p in model.parameters())


def format_number(num: Union[int, float], precision: int = 2) -> str:
  """
  Format a number with appropriate suffixes (K, M, B).

  Args:
      num: Number to format
      precision: Decimal precision

  Returns:
      Formatted string
  """
  if num >= 1e9:
    return f"{num / 1e9:.{precision}f}B"
  elif num >= 1e6:
    return f"{num / 1e6:.{precision}f}M"
  elif num >= 1e3:
    return f"{num / 1e3:.{precision}f}K"
  else:
    return f"{num:.{precision}f}"


def ensure_dir(directory: str) -> str:
  """
  Ensure a directory exists, creating it if necessary.

  Args:
      directory: Directory path

  Returns:
      The directory path
  """
  os.makedirs(directory, exist_ok=True)
  return directory


def get_model_size_mb(model: torch.nn.Module) -> float:
  """
  Get the size of a model in megabytes.

  Args:
      model: PyTorch model

  Returns:
      Model size in MB
  """
  param_size = 0
  buffer_size = 0

  for param in model.parameters():
    param_size += param.nelement() * param.element_size()

  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_mb = (param_size + buffer_size) / 1024 / 1024
  return size_mb


def log_system_info() -> None:
  """Log system and environment information."""
  logger = logging.getLogger(__name__)

  logger.info("=== System Information ===")
  logger.info(f"PyTorch version: {torch.__version__}")

  if torch.cuda.is_available():
    try:
      # Try to get CUDA version info
      logger.info("CUDA is available")
    except Exception:
      logger.info("CUDA version: Unknown")
    logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
      logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
      props = torch.cuda.get_device_properties(i)
      logger.info(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
  else:
    logger.info("CUDA not available")

  logger.info("=" * 25)


class ProgressMeter:
  """Simple progress meter for tracking training progress."""

  def __init__(self, num_batches: int, meters: list, prefix: str = ""):
    """
    Initialize progress meter.

    Args:
        num_batches: Total number of batches
        meters: List of meters to track
        prefix: Prefix for log messages
    """
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch: int) -> None:
    """Display current progress."""
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    logging.info("\t".join(entries))

  def _get_batch_fmtstr(self, num_batches: int) -> str:
    """Get format string for batch numbers."""
    num_digits = len(str(num_batches // 1))
    fmt = "{:" + str(num_digits) + "d}"
    return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
  """
  Unwrap a model from DDP/DataParallel wrapper.

  Args:
      model: Potentially wrapped model

  Returns:
      Unwrapped model
  """
  if hasattr(model, "module"):
    return model.module
  else:
    return model
