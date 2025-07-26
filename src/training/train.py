"""
Training utilities for CLIP-Self models.

This module provides training loop functionality, evaluation utilities,
and helper functions for CLIP-based self-supervised learning.
"""

import json
import logging
import math
import os
import time
from typing import Any, Dict, Optional

import torch

from open_clip import get_cast_dtype
from training.distributed import is_master
from training.precision import get_autocast
from training.zero_shot import zero_shot_eval


class AverageMeter:
  """
  Computes and stores the average and current value of metrics during training.

  This class is useful for tracking loss values, accuracy, and other metrics
  that need to be averaged over multiple batches or epochs.

  Attributes:
    val (float): Current value
    avg (float): Running average
    sum (float): Sum of all values
    count (int): Number of updates
  """

  def __init__(self) -> None:
    """Initialize the meter with zero values."""
    self.reset()

  def reset(self) -> None:
    """Reset all stored values to zero."""
    self.val: float = 0.0
    self.avg: float = 0.0
    self.sum: float = 0.0
    self.count: int = 0

  def update(self, val: float, n: int = 1) -> None:
    """
    Update the meter with a new value.

    Args:
      val (float): The new value to add
      n (int): The weight/count for this value (default: 1)
    """
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def postprocess_clip_output(model_out: Any) -> Dict[str, Any]:
  """
  Post-process CLIP model output into a structured dictionary.

  Args:
    model_out: Raw model output tuple containing image features, text features, and logit scale

  Returns:
    Dictionary with keys: 'image_features', 'text_features', 'logit_scale'
  """
  return {"image_features": model_out[0], "text_features": model_out[1], "logit_scale": model_out[2]}


def unwrap_model(model: Any) -> Any:
  """
  Unwrap a model from its distributed wrapper if present.

  Args:
    model: The model, potentially wrapped with DataParallel or DistributedDataParallel

  Returns:
    The unwrapped model
  """
  if hasattr(model, "module"):
    return model.module
  else:
    return model


def backward(total_loss: torch.Tensor, scaler: Optional[torch.GradScaler]) -> None:
  """
  Perform backward pass with optional gradient scaling.

  Args:
    total_loss (torch.Tensor): The computed loss tensor
    scaler (torch.GradScaler, optional): Gradient scaler for mixed precision training (can be None)
  """
  if scaler is not None:
    scaler.scale(total_loss).backward()
  else:
    total_loss.backward()


@torch.no_grad()
def student_teacher_ensemble(
  student: Dict[str, torch.Tensor], teacher: Dict[str, torch.Tensor], alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
  """
  Create an ensemble of student and teacher model state dictionaries.

  This function performs exponential moving average (EMA) between student and teacher weights.

  Args:
    student: Student model state dictionary
    teacher: Teacher model state dictionary
    alpha (float): Interpolation factor (0.0 = full teacher, 1.0 = full student)

  Returns:
      Target state dictionary with interpolated weights
  """
  target_state_dict: Dict[str, Any] = {}
  for k, v in student.items():
    target_state_dict[k] = v * alpha + teacher[k] * (1.0 - alpha)

  return target_state_dict


def train_one_epoch(
  model: Any,
  method: Any,
  data: Dict[str, Any],
  loss: Any,
  epoch: int,
  optimizer: Any,
  scaler: Any,
  scheduler: Any,
  dist_model: Any,
  args: Any,
) -> None:
  """
  Train the model for one epoch.

  This function performs one complete pass through the training dataset,
  including forward pass, loss computation, backward pass, and parameter updates.

  Args:
    model: The model to train
    method: Training method/strategy function
    data: Dictionary containing training data loaders
    loss: Loss function
    epoch: Current epoch number
    optimizer: Optimizer for parameter updates
    scaler: Gradient scaler for mixed precision training
    scheduler: Learning rate scheduler
    dist_model: Distributed model (teacher model for distillation, can be None)
    args: Training arguments/configuration
  """
  # Setup training environment
  device = torch.device(args.device)
  autocast = get_autocast(args.precision)
  cast_dtype = get_cast_dtype(args.precision)

  # Set model modes
  model.train()
  if dist_model is not None:
    dist_model.eval()

  # Prepare data loader for this epoch
  data["train"].set_epoch(epoch)  # Set epoch in process safe manner via sampler or shared_epoch
  dataloader = data["train"].dataloader
  num_batches_per_epoch = dataloader.num_batches // args.accum_freq
  sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

  # Initialize metrics tracking
  losses_m: Dict[str, Any] = {}  # Dictionary to store loss meters
  batch_time_m = AverageMeter()
  data_time_m = AverageMeter()

  end: float = time.time()

  # Main training loop
  for i, batch in enumerate(dataloader):
    i_accum = i // args.accum_freq
    step = num_batches_per_epoch * epoch + i_accum

    # Update learning rate
    if not args.skip_scheduler:
      scheduler(step)

    # Track data loading time
    data_time_m.update(time.time() - end)
    optimizer.zero_grad()

    # Gradient accumulation check (currently disabled)
    assert args.accum_freq == 1, "accum freq disabled"

    # Forward pass with automatic mixed precision
    with autocast():
      losses, batch_size, logit_scale = method(
        batch, model, dist_model, loss, device, cast_dtype, args.distributed, args
      )
      total_loss: int = sum(losses.values())
      losses["loss"] = total_loss

    # Backward pass
    backward(total_loss, scaler)

    # Optimizer step with gradient clipping
    if scaler is not None:
      if args.horovod:
        # Horovod-specific optimization
        optimizer.synchronize()
        scaler.unscale_(optimizer)
        if args.grad_clip_norm is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        with optimizer.skip_synchronize():
          scaler.step(optimizer)
      else:
        # Standard mixed precision optimization
        if args.grad_clip_norm is not None:
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        scaler.step(optimizer)
      scaler.update()
    else:
      # Standard optimization without scaler
      if args.grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
      optimizer.step()

    # Clamp logit scale as per CLIP paper (ln(100) = 4.6052)
    with torch.no_grad():
      unwrap_model(model).logit_scale.clamp_(0, math.log(100))

    # Track batch processing time
    batch_time_m.update(time.time() - end)
    end = time.time()
    batch_count = i_accum + 1

    # Logging and metrics tracking
    if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
      # Calculate training statistics
      num_samples = batch_count * batch_size * args.accum_freq * args.world_size
      samples_per_epoch = dataloader.num_samples
      percent_complete = 100.0 * batch_count / num_batches_per_epoch

      # Update loss meters (only on master node for efficiency)
      for key, val in losses.items():
        if key not in losses_m:
          losses_m[key] = AverageMeter()
        losses_m[key].update(val.item(), batch_size)

      # Prepare logging information
      logit_scale_scalar = logit_scale.item()
      loss_log = " ".join(
        [f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" for loss_name, loss_m in losses_m.items()]
      )
      samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
      samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val

      # Log training progress
      logging.info(
        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
        f"Data (t): {data_time_m.avg:.3f} "
        f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
        f"LR: {optimizer.param_groups[0]['lr']:5f} "
        f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
      )

      # Prepare additional logging data
      log_data: Dict[str, Any] = {
        "data_time": data_time_m.val,
        "batch_time": batch_time_m.val,
        "samples_per_second": samples_per_second,
        "samples_per_second_per_gpu": samples_per_second_per_gpu,
        "scale": logit_scale_scalar,
        "lr": optimizer.param_groups[0]["lr"],
      }
      log_data.update({name: val.val for name, val in losses_m.items()})

      # Reset meters for next logging window
      batch_time_m.reset()
      data_time_m.reset()


def evaluate(model: Any, data: Dict[str, Any], epoch: int, args: Any) -> Dict[str, Any]:
  """
  Evaluate the model on validation/test datasets.

  This function runs the model in evaluation mode and computes various metrics
  including zero-shot classification performance.

  Args:
    model: The model to evaluate
    data: Dictionary containing evaluation data loaders
    epoch: Current epoch number for logging
    args: Evaluation arguments/configuration

  Returns:
      Dictionary containing evaluation metrics
  """
  metrics: Dict[str, Any] = {}
  model.eval()

  # Perform zero-shot evaluation
  zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

  # Only master process handles metric collection and logging
  if not is_master(args):
    return {}

  metrics.update(zero_shot_metrics)
  if not metrics:
    return metrics

  # Format metrics for logging (only those with 'all' in the key)
  keys: str = "".join([f"{k}, " for k in metrics.keys() if "all" in k])[:-2]
  values: str = "".join([f"{round(v, 4):.4f}, " for k, v in metrics.items() if "all" in k])[:-2]

  # Log evaluation results
  logging.info(f"Eval Epoch: {epoch}. " + f"{keys}: {values}.")
  # TODO: save the results as plots
  logging.info(metrics)

  # Save metrics to file if requested
  if args.save_logs:
    with open(os.path.join(args.checkpoint_path, "results.json"), "a+") as f:
      f.write(json.dumps(metrics))
      f.write("\n")

  return metrics
