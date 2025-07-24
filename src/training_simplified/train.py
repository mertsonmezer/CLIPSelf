from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Any, Callable, Dict

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from open_clip import get_cast_dtype
from training_simplified.data import DataInfo
from training_simplified.utils.distributed import is_master
from training_simplified.utils.precision import get_autocast
from training_simplified.utils.running_avg_tracker import RunningAverageTracker
from training_simplified.utils.zero_shot import zero_shot_eval


def interpolate_model_weights(
  student_weights: Dict[str, Tensor],
  teacher_weights: Dict[str, Tensor],
  interpolation_factor: float = 0.5,
) -> Dict[str, Tensor]:
  """
  Create an ensemble of student and teacher model weights through interpolation.

  This function combines two sets of model weights using linear interpolation,
  commonly used in knowledge distillation or model ensembling techniques.

  Args:
    student_weights: State dict from the student model
    teacher_weights: State dict from the teacher model
    interpolation_factor: Weight for student model (0.0 = full teacher, 1.0 = full student)

  Returns:
      Dictionary containing interpolated weights for each parameter
  """
  teacher_weight: float = 1.0 - interpolation_factor

  return {
    param_name: student_weights[param_name] * interpolation_factor + teacher_weights[param_name] * teacher_weight
    for param_name in student_weights
  }


def train_one_epoch(
  student_model,
  training_method: Callable[..., Any],
  dataloaders: Dict[str, DataInfo],
  loss_fn,
  current_epoch: int,
  optimizer,
  gradient_scaler,
  lr_scheduler,
  teacher_model,
  args: argparse.Namespace,
) -> None:
  """
  Execute one complete training epoch with knowledge distillation.

  This function handles the main training loop including:
  - Forward/backward passes
  - Loss computation and backpropagation
  - Learning rate scheduling
  - Gradient scaling (for mixed precision)
  - Knowledge distillation from teacher to student
  - Logging and monitoring

  Args:
    student_model: The model being trained (learns from data and teacher)
    training_method: Function that defines how to process each batch
    dataset_loaders: Dictionary containing train/val data loaders
    loss_function: Loss function to optimize
    current_epoch: Current epoch number (0-indexed)
    optimizer: PyTorch optimizer for student model
    gradient_scaler: For mixed precision training (None if not used)
    learning_rate_scheduler: Learning rate scheduler (None if not used)
    teacher_model: Pre-trained model providing knowledge distillation (None if not used)
    args: Configuration arguments
  """
  # Setup training environment
  device = torch.device(args.device)
  autocast_context: Callable[..., Any] | Any = get_autocast(args.precision)
  tensor_dtype: torch.dtype | None = get_cast_dtype(args.precision)

  # Set model modes: training for student, eval for teacher
  student_model.train()
  if teacher_model is not None:
    teacher_model.eval()  # Teacher stays in eval mode during distillation

  # Prepare data loader and set epoch for distributed training
  train_dataloader: DataLoader[Any] = dataloaders["train"].dataloader
  dataloaders["train"].set_epoch(current_epoch)

  # Initialize metrics tracking
  loss_tracker = RunningAverageTracker()
  batch_processing_time = RunningAverageTracker()
  data_loading_time = RunningAverageTracker()

  # Main training loop
  epoch_start_time: float = time.time()

  for idx, data in enumerate(train_dataloader):
    if lr_scheduler is not None:
      global_step: int = idx + current_epoch * len(train_dataloader)
      lr_scheduler(global_step)

    data_loading_time.update(time.time() - epoch_start_time)

    # Reset gradients for student model
    optimizer.zero_grad()

    with autocast_context():
      loss_components, batch_size, logit_scale_value = training_method(
        data,
        student_model,
        teacher_model,
        # loss_fn,
        device,
        tensor_dtype,
        args.distributed,
        args,
      )

      # Sum all loss components (e.g., contrastive loss, distillation loss, etc.)
      total_loss = sum(loss_components.values())

    # Backward pass with gradient scaling if using mixed precision
    if gradient_scaler is not None:
      gradient_scaler.scale(total_loss).backward()
      gradient_scaler.step(optimizer)
      gradient_scaler.update()
    else:
      total_loss.backward()
      optimizer.step()

    # Clamp logit scale to prevent numerical instability
    # CLIP models use a learnable temperature parameter that needs bounds
    with torch.no_grad():
      student_model.logit_scale.clamp_(0, math.log(100))

    # Update metrics
    loss_tracker.update(total_loss.item(), batch_size)
    batch_processing_time.update(time.time() - epoch_start_time)
    epoch_start_time = time.time()

    # Periodic logging
    if is_master(args) and idx % args.log_every_n_steps == 0:
      logging.info(
        f"Train Epoch: {current_epoch} [{idx:4d}/{len(train_dataloader):4d}] "
        f"Loss: {loss_tracker.running_average:.4f} "
        f"Logit Scale: {logit_scale_value.item():.3f} "
        f"Batch Time: {batch_processing_time.running_average:.3f}s "
        f"Data Time: {data_loading_time.running_average:.3f}s"
      )


def run_zero_shot_evaluation(
  student_model, dataloaders: Dict[str, DataInfo], current_epoch: int, args: argparse.Namespace
) -> Dict[Any, Any]:
  """
  Perform zero-shot evaluation on the validation dataset.

  This function switches the student model to evaluation mode and runs zero-shot
  classification on the validation set to monitor training progress.

  Args:
    student_model: The trained student model to evaluate
    dataset_loaders: Dictionary containing validation data loaders
    current_epoch: Current epoch number for logging
    training_args: Configuration arguments

  Returns:
    Dict of evaluation metrics (accuracy, etc.) or None if evaluation fails
  """
  student_model.eval()

  # Run zero-shot evaluation (defined in utils)
  evaluation_metrics: Dict[Any, Any] = zero_shot_eval(student_model, dataloaders, current_epoch, args)

  # Log results if we're on the master process and have valid metrics
  if is_master(args) and evaluation_metrics:
    metric_names: str = ", ".join(evaluation_metrics.keys())
    metric_values: str = ", ".join(f"{value:.4f}" for value in evaluation_metrics.values())
    logging.info(f"Evaluation Epoch: {current_epoch} | {metric_names}: {metric_values}")

  return evaluation_metrics
