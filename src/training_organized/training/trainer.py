"""
Main trainer class for CLIPSelf.

This module provides a clean, well-structured trainer that orchestrates the entire training process.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.modules.module import Module

from training.data import DataInfo
from training_organized.clipself import CLIPSelfMethod
from training_organized.config import CLIPSelfFullConfig


class AverageMeter:
  """Computes and stores the average and current value."""

  def __init__(self, name: str = ""):
    self.name: str = name
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val: float, n: int = 1):
    self.val: float = val
    self.sum += val * n
    self.count += n
    self.avg: float = self.sum / self.count

  def __str__(self) -> str:
    return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class CLIPSelfTrainer:
  """
  Main trainer for CLIPSelf method.

  This class handles the training loop, model management, and logging.
  """

  def __init__(
    self,
    config: CLIPSelfFullConfig,
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    clipself_method: CLIPSelfMethod,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    scaler: Optional[GradScaler] = None,
  ):
    """
    Initialize the trainer.

    Args:
      config (CLIPSelfFullConfig): Full configuration
      model (torch.nn.Module): Student model to train
      teacher_model (torch.nn.Module): Teacher model (frozen)
      clipself_method (CLIPSelfMethod): CLIPSelf method instance
      optimizer (torch.optim.Optimizer): Optimizer for training
      scheduler (Optional[object]): Learning rate scheduler
      scaler (Optional[GradScaler]): Gradient scaler for mixed precision
    """
    self.config: CLIPSelfFullConfig = config
    self.model: Module = model
    self.teacher_model: Module = teacher_model
    self.clipself_method: CLIPSelfMethod = clipself_method
    self.optimizer: optim.Optimizer = optimizer
    self.scheduler: object | None = scheduler
    self.scaler: GradScaler | None = scaler

    self.device: torch.device = torch.device(config.device)
    self.logger: logging.Logger = logging.getLogger(__name__)

    # Training state
    self.current_epoch: int = 0
    self.global_step: int = 0

    # Metrics tracking
    self.loss_meter: AverageMeter = AverageMeter("Loss")
    self.cosine_meter: AverageMeter = AverageMeter("Cosine")
    self.batch_time_meter: AverageMeter = AverageMeter("Batch Time")
    self.data_time_meter: AverageMeter = AverageMeter("Data Time")

  def train_one_epoch(self, data_info: DataInfo, epoch: int) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        dataloader: Training data loader
        epoch: Current epoch number

    Returns:
        Dictionary of training metrics
    """
    self.current_epoch = epoch
    self.model.train()

    self.teacher_model.eval()

    # Reset meters
    self.loss_meter.reset()
    self.cosine_meter.reset()
    self.batch_time_meter.reset()
    self.data_time_meter.reset()

    # Set epoch for distributed sampler
    if hasattr(data_info.sampler, "set_epoch"):
      data_info.sampler.set_epoch(epoch)

    dataloader = data_info.dataloader
    num_batches = len(dataloader)
    self.logger.info(f"Starting epoch {epoch} with {num_batches} batches")

    end_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
      # Measure data loading time
      self.data_time_meter.update(time.time() - end_time)

      # Training step
      step_metrics: Dict[str, float] = self._train_step(batch, batch_idx, num_batches)

      # Update meters
      self.loss_meter.update(step_metrics["loss"], len(batch[0]))
      if "cosine_similarity" in step_metrics:
        self.cosine_meter.update(step_metrics["cosine_similarity"], len(batch[0]))

      # Measure batch time
      self.batch_time_meter.update(time.time() - end_time)

      # Logging
      if batch_idx % self._get_log_frequency(num_batches) == 0:
        self._log_training_progress(batch_idx, num_batches)

      end_time: float = time.time()

    # Epoch summary
    epoch_metrics: Dict[str, Any] = {
      "epoch": epoch,
      "loss": self.loss_meter.avg,
      "cosine_similarity": self.cosine_meter.avg,
      "batch_time": self.batch_time_meter.avg,
      "data_time": self.data_time_meter.avg,
    }

    self.logger.info(f"Epoch {epoch} completed. Loss: {self.loss_meter.avg:.4f}, Cosine: {self.cosine_meter.avg:.4f}")

    return epoch_metrics

  def _train_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int, num_batches: int) -> Dict[str, float]:
    """
    Perform one training step.

    Args:
      batch: Input batch
      batch_idx: Current batch index
      num_batches: Total number of batches

    Returns:
      Dictionary of step metrics
    """
    # Update learning rate scheduler
    if self.scheduler is not None and not self.config.training.skip_scheduler:
      step: int = num_batches * self.current_epoch + batch_idx
      self.scheduler(step)

    # Zero gradients
    self.optimizer.zero_grad()

    # Get precision settings
    cast_dtype = self._get_cast_dtype()

    # Forward pass with mixed precision
    with self._get_autocast():
      losses, batch_size, logit_scale = self.clipself_method(
        batch=batch,
        student_model=self.model,
        teacher_model=self.teacher_model,
        device=self.device,
        cast_dtype=cast_dtype,
        distributed=self.config.training.distributed,
      )

      total_loss: torch.Tensor = losses["loss_cosine"]

    # Backward pass
    self._backward(total_loss)

    # Optimizer step
    self._optimizer_step()

    # Prepare metrics for return
    step_metrics: Dict[str, Any] = {
      "loss": total_loss.item(),
      "logit_scale": logit_scale.item(),
      "batch_size": batch_size,
    }

    # Add additional metrics if available
    if "cosine_similarity" in losses:
      step_metrics["cosine_similarity"] = losses["cosine_similarity"].item()

    self.global_step += 1
    return step_metrics

  def _backward(self, loss: torch.Tensor):
    """Perform backward pass with optional gradient scaling."""
    if self.scaler is not None:
      self.scaler.scale(loss).backward()
    else:
      loss.backward()

  def _optimizer_step(self):
    """Perform optimizer step with optional gradient scaling."""
    if self.scaler is not None:
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      self.optimizer.step()

  def _get_cast_dtype(self) -> Optional[torch.dtype]:
    """Get the appropriate dtype for mixed precision."""
    precision = self.config.model.precision
    if precision == "amp":
      return torch.float16
    elif precision == "fp16":
      return torch.float16
    else:
      return None

  def _get_autocast(self):
    """Get autocast context for mixed precision."""
    if self.config.model.precision == "amp":
      return torch.cuda.amp.autocast()
    else:
      # No-op context manager
      from contextlib import nullcontext

      return nullcontext()

  def _get_log_frequency(self, num_batches: int) -> int:
    """Calculate logging frequency based on number of batches."""
    if num_batches < 10:
      return 1
    elif num_batches < 100:
      return 10
    else:
      return max(1, num_batches // 10)

  def _log_training_progress(self, batch_idx: int, num_batches: int):
    """Log training progress."""
    progress = 100.0 * batch_idx / num_batches
    self.logger.info(
      f"Epoch {self.current_epoch} [{batch_idx:4d}/{num_batches}] ({progress:5.1f}%) | "
      f"{self.loss_meter} | {self.cosine_meter} | "
      f"Batch Time: {self.batch_time_meter.val:.3f}s | "
      f"Data Time: {self.data_time_meter.val:.3f}s"
    )

  def get_state_dict(self) -> Dict[str, Any]:
    """Get trainer state for checkpointing."""
    state: Dict[str, Any] = {
      "epoch": self.current_epoch,
      "global_step": self.global_step,
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

    if self.scaler is not None:
      state["scaler_state_dict"] = self.scaler.state_dict()

    return state

  def load_state_dict(self, state: Dict[str, Any]):
    """Load trainer state from checkpoint."""
    self.current_epoch = state.get("epoch", 0)
    self.global_step = state.get("global_step", 0)

    self.model.load_state_dict(state["model_state_dict"])
    self.optimizer.load_state_dict(state["optimizer_state_dict"])

    if self.scaler is not None and "scaler_state_dict" in state:
      self.scaler.load_state_dict(state["scaler_state_dict"])

    self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_clipself_trainer(
  config: CLIPSelfFullConfig,
  model: torch.nn.Module,
  teacher_model: torch.nn.Module,
  clipself_method: CLIPSelfMethod,
  optimizer: optim.Optimizer,
  scheduler: Optional[object] = None,
  scaler: Optional[GradScaler] = None,
) -> CLIPSelfTrainer:
  """
  Factory function to create CLIPSelf trainer.

  Args:
    config: Full configuration
    model: Student model
    teacher_model: Teacher model
    clipself_method: CLIPSelf method
    optimizer: Optimizer
    scheduler: Learning rate scheduler
    scaler: Gradient scaler

  Returns:
    CLIPSelf trainer instance
  """
  return CLIPSelfTrainer(
    config=config,
    model=model,
    teacher_model=teacher_model,
    clipself_method=clipself_method,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
  )
