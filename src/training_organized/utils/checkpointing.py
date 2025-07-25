"""
Checkpointing utilities for CLIPSelf training.
"""

import os
import logging
from typing import Dict, Any, Optional

import torch


def save_checkpoint(
  state: Dict[str, Any],
  checkpoint_dir: str,
  filename: str = "checkpoint.pt",
  is_best: bool = False,
  save_latest: bool = True,
) -> str:
  """
  Save training checkpoint.

  Args:
      state: State dictionary to save
      checkpoint_dir: Directory to save checkpoints
      filename: Checkpoint filename
      is_best: Whether this is the best checkpoint
      save_latest: Whether to also save as latest checkpoint

  Returns:
      Path to saved checkpoint
  """
  logger = logging.getLogger(__name__)

  # Create checkpoint directory
  os.makedirs(checkpoint_dir, exist_ok=True)

  # Save main checkpoint
  checkpoint_path = os.path.join(checkpoint_dir, filename)
  torch.save(state, checkpoint_path)
  logger.info(f"Checkpoint saved: {checkpoint_path}")

  # Save as best if specified
  if is_best:
    best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    torch.save(state, best_path)
    logger.info(f"Best checkpoint saved: {best_path}")

  # Save as latest if specified
  if save_latest:
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(state, latest_path)
    logger.info(f"Latest checkpoint saved: {latest_path}")

  return checkpoint_path


def load_checkpoint(checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
  """
  Load training checkpoint.

  Args:
      checkpoint_path: Path to checkpoint file
      map_location: Device to map tensors to

  Returns:
      Loaded state dictionary
  """
  logger = logging.getLogger(__name__)

  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  logger.info(f"Loading checkpoint: {checkpoint_path}")

  try:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")

    # Log checkpoint info if available
    if "epoch" in checkpoint:
      logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "global_step" in checkpoint:
      logger.info(f"Checkpoint global step: {checkpoint['global_step']}")

    return checkpoint

  except Exception as e:
    logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
    raise


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
  """
  Find the latest checkpoint in a directory.

  Args:
      checkpoint_dir: Directory to search for checkpoints

  Returns:
      Path to latest checkpoint or None if not found
  """
  if not os.path.exists(checkpoint_dir):
    return None

  # Look for latest_checkpoint.pt first
  latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
  if os.path.exists(latest_path):
    return latest_path

  # Look for numbered epoch checkpoints
  checkpoint_files = []
  for filename in os.listdir(checkpoint_dir):
    if filename.startswith("epoch_") and filename.endswith(".pt"):
      checkpoint_files.append(filename)

  if not checkpoint_files:
    return None

  # Sort by epoch number and return the latest
  checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
  return os.path.join(checkpoint_dir, checkpoint_files[-1])


def cleanup_old_checkpoints(checkpoint_dir: str, keep_latest: int = 5, keep_best: bool = True) -> None:
  """
  Clean up old checkpoints, keeping only the most recent ones.

  Args:
      checkpoint_dir: Directory containing checkpoints
      keep_latest: Number of latest checkpoints to keep
      keep_best: Whether to keep the best checkpoint
  """
  logger = logging.getLogger(__name__)

  if not os.path.exists(checkpoint_dir):
    return

  # Get all epoch checkpoint files
  epoch_files = []
  for filename in os.listdir(checkpoint_dir):
    if filename.startswith("epoch_") and filename.endswith(".pt"):
      epoch_files.append(filename)

  if len(epoch_files) <= keep_latest:
    return  # Nothing to clean up

  # Sort by epoch number
  epoch_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

  # Files to remove (keep the latest ones)
  files_to_remove = epoch_files[:-keep_latest]

  # Don't remove best checkpoint if it exists
  best_checkpoint = "best_checkpoint.pt"
  if keep_best and best_checkpoint in files_to_remove:
    files_to_remove.remove(best_checkpoint)

  # Remove old checkpoints
  for filename in files_to_remove:
    file_path = os.path.join(checkpoint_dir, filename)
    try:
      os.remove(file_path)
      logger.info(f"Removed old checkpoint: {filename}")
    except OSError as e:
      logger.warning(f"Failed to remove checkpoint {filename}: {e}")


class CheckpointManager:
  """Manager for handling checkpoints throughout training."""

  def __init__(self, checkpoint_dir: str, keep_latest: int = 5):
    """
    Initialize checkpoint manager.

    Args:
        checkpoint_dir: Directory to store checkpoints
        keep_latest: Number of latest checkpoints to keep
    """
    self.checkpoint_dir = checkpoint_dir
    self.keep_latest = keep_latest
    self.logger = logging.getLogger(__name__)

    os.makedirs(checkpoint_dir, exist_ok=True)

  def save(self, state: Dict[str, Any], epoch: int, is_best: bool = False) -> str:
    """Save checkpoint for given epoch."""
    filename = f"epoch_{epoch:04d}.pt"
    return save_checkpoint(
      state=state, checkpoint_dir=self.checkpoint_dir, filename=filename, is_best=is_best, save_latest=True
    )

  def load_latest(self) -> Optional[Dict[str, Any]]:
    """Load the latest checkpoint."""
    latest_path = find_latest_checkpoint(self.checkpoint_dir)
    if latest_path is None:
      return None
    return load_checkpoint(latest_path)

  def cleanup(self) -> None:
    """Clean up old checkpoints."""
    cleanup_old_checkpoints(self.checkpoint_dir, keep_latest=self.keep_latest, keep_best=True)
