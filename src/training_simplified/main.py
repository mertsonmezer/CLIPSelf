"""
CLIPSelf Training Script - Simplified Version

This module provides the main training pipeline for CLIPSelf, a self-supervised learning
method for CLIP models. It handles model initialization, data loading, training loop,
and checkpoint management.

Key components:
- Model setup with pretrained weights
- Data preprocessing and loading
- Training loop with automatic mixed precision
- Checkpoint saving and resuming
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.cuda.amp import GradScaler

from open_clip import create_model, create_model_and_transforms
from open_clip.coca_model import CoCa
from open_clip.model import CLIP, CustomTextCLIP
from training_simplified.clipself import CLIPSelf
from training_simplified.data import DataInfo, get_data
from training_simplified.train import train_one_epoch

# The model type is determined by the factory logic:
CLIPType = Union[CLIP, CustomTextCLIP, CoCa]


def set_random_seed(seed_value: int) -> None:
  """
  Sets random seeds for reproducible training across different libraries.

  Args:
    seed_value (int): The seed value to use for random number generators
  """
  torch.manual_seed(seed_value)
  np.random.seed(seed_value)
  random.seed(seed_value)


def setup_logging_and_directories(args: argparse.Namespace) -> str:
  """
  Creates log directory and setup logging configuration.

  Args:
    args (argparse.Namespace): Parsed command line arguments

  Returns:
    str: Path to the created log directory
  """
  # Generate experiment name if not provided
  if args.name is None:
    timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = f"{args.model}-{timestamp}"

  # Create log directory
  log_directory: str = os.path.join(args.logs, args.name)
  os.makedirs(log_directory, exist_ok=True)

  # Setup logging to file
  log_file_path: str = os.path.join(log_directory, "training.log")
  logging.basicConfig(
    level=logging.INFO,
    filename=log_file_path,
    format="%(asctime)s - %(levelname)s - %(message)s",
  )

  logging.info(f"Starting training experiment: {args.name}")
  logging.info(f"Log directory: {log_directory}")

  return log_directory


def initialize_models(
  args: argparse.Namespace, device: Literal["cuda", "cpu"]
) -> Tuple[CLIPType, T.Compose, T.Compose, CLIPType]:
  """
  Initializes the student and teacher models.

  Args:
    args (argparse.Namespace): Parsed command line arguments.
    device (Literal["cuda", "cpu"]): Device to use for training ("cuda" or "cpu")

  Returns:
    Tuple: (student_model, preprocess_train, preprocess_val, teacher_model)
  """
  logging.info(f"Initializing models on device: {device}")

  # Create main training model with preprocessing transforms
  student_model, preprocess_train, preprocess_val = create_model_and_transforms(
    args.model,
    args.pretrained,
    precision=args.precision,
    device=device,
    output_dict=True,
    cache_dir=args.cache_dir,
    dataset_type=args.dataset_type,
  )

  # Create distance model for evaluation (kept in eval mode)
  teacher_model = create_model(
    args.model,
    args.pretrained,
    device=device,
    precision=args.precision,
    output_dict=True,
    cache_dir=args.cache_dir,
  )
  teacher_model.eval()

  logging.info("Models initialized successfully")
  return student_model, preprocess_train, preprocess_val, teacher_model


def setup_training_components(
  student_model: CLIPType, args: argparse.Namespace
) -> Tuple[optim.Optimizer, None, GradScaler | None]:
  """
  Setups optimizer, scheduler, and gradient scaler for training.

  Args:
    student_model (CLIPType): The model to train
    args (argparse.Namespace): Parsed command line arguments

  Returns:
    Tuple: (optimizer, lr_scheduler, gradient_scaler)
  """
  # Initialize optimizer with specified hyperparameters
  optimizer = optim.AdamW(
    student_model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
    weight_decay=args.wd,
  )

  lr_scheduler = None  # No scheduler in simplified version

  # Setup gradient scaler for automatic mixed precision training
  gradient_scaler: GradScaler | None = GradScaler() if args.precision.startswith("amp") else None

  logging.info(f"Optimizer: AdamW with lr={args.lr}, wd={args.wd}")
  logging.info(f"Mixed precision: {args.precision}")

  return optimizer, lr_scheduler, gradient_scaler


def load_checkpoint_if_resuming(
  student_model: CLIPType, optimizer: optim.Optimizer, gradient_scaler: GradScaler | None, args: argparse.Namespace
) -> int:
  """
  Loads checkpoint if resuming training from a previous run.

  Args:
    student_model: The model to load weights into
    optimizer: The optimizer to load state into
    gradient_scaler: The gradient scaler to load state into
    args: Parsed command line arguments

  Returns:
    int: Starting epoch number
  """
  starting_epoch = 0

  if args.resume:
    logging.info(f"Resuming training from checkpoint: {args.resume}")

    checkpoint = torch.load(args.resume, map_location="cpu")
    model_state = checkpoint.get("state_dict", checkpoint)

    # Load model weights
    student_model.load_state_dict(model_state)

    # Load optimizer state if available
    if "optimizer" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer"])

    # Load gradient scaler state if available
    if gradient_scaler and "scaler" in checkpoint:
      gradient_scaler.load_state_dict(checkpoint["scaler"])

    # Get starting epoch
    starting_epoch: int = checkpoint.get("epoch", 0)
    logging.info(f"Resumed from epoch {starting_epoch}")

  return starting_epoch


def save_training_checkpoint(
  student_model, optimizer: optim.Optimizer, gradient_scaler: GradScaler, epoch_num: int, log_directory: str
):
  """
  Save training checkpoint with model, optimizer, and scaler states.

  Args:
    student_model (CLIPType): The trained model
    optimizer (optim.Optimizer): The optimizer state
    gradient_scaler (GradScaler): The gradient scaler state
    epoch_num (int): Current epoch number
    log_directory (str): Directory to save checkpoint
  """
  checkpoint_data: Dict[str, Any] = {
    "epoch": epoch_num + 1,
    "state_dict": student_model.state_dict(),
    "optimizer": optimizer.state_dict(),
  }

  # Add gradient scaler state if using mixed precision
  if gradient_scaler:
    checkpoint_data["scaler"] = gradient_scaler.state_dict()

  checkpoint_path: str = os.path.join(log_directory, f"epoch_{epoch_num + 1}.pt")
  torch.save(checkpoint_data, checkpoint_path)
  logging.info(f"Checkpoint saved: {checkpoint_path}")


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
  """
  Parse command line arguments for CLIPSelf training.

  Args:
    args: Optional list of arguments to parse (defaults to sys.argv)

  Returns:
    Parsed arguments namespace
  """
  parser = argparse.ArgumentParser(description="CLIPSelf Training")

  # Model and checkpoint arguments
  parser.add_argument("--model", type=str, default="ViT-B-16", help="CLIP model architecture")
  parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b88k", help="Pretrained weights")
  parser.add_argument(
    "--precision", type=str, default="fp32", choices=["fp32", "fp16", "amp"], help="Training precision"
  )
  parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for models")
  parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

  # Dataset arguments
  parser.add_argument(
    "--dataset_type",
    type=str,
    default="grid_distill",
    choices=["grid_distill", "proposals_distill"],
    help="Dataset type",
  )
  parser.add_argument("--train_data", type=str, required=True, help="Path to training annotations JSON")
  parser.add_argument("--train_image_root", type=str, required=True, help="Path to training images directory")
  parser.add_argument("--val_data", type=str, default=None, help="Path to validation annotations JSON")
  parser.add_argument("--val_image_root", type=str, default=None, help="Path to validation images directory")
  parser.add_argument("--max_split", type=int, default=4, help="Maximum grid divisions per dimension")
  parser.add_argument("--input_size", type=int, default=224, help="Input image size")
  parser.add_argument("--max_boxes", type=int, default=20, help="Maximum number of boxes per image")

  # Training arguments
  parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
  parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
  parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
  parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
  parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
  parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
  parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
  parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")

  # CLIPSelf specific arguments
  parser.add_argument("--cosine_weight", type=float, default=1.0, help="Weight for cosine loss")
  parser.add_argument("--extract_type", type=str, default="roi_align", help="Feature extraction type")
  parser.add_argument("--multiscale", action="store_true", help="Enable multiscale augmentation")

  # Logging and misc arguments
  parser.add_argument("--name", type=str, default=None, help="Experiment name")
  parser.add_argument("--logs", type=str, default="./logs", help="Log directory")
  parser.add_argument("--seed", type=int, default=42, help="Random seed")
  parser.add_argument("--log_every_n_steps", type=int, default=100, help="Log frequency")

  # Distributed training arguments
  parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
  parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed training")
  parser.add_argument("--local_rank", type=int, default=0, help="Local process rank for distributed training")

  return parser.parse_args(args)


def main(command_line_args: List[str] | None = None) -> None:
  args: argparse.Namespace = parse_args(command_line_args)

  # Setup device and CUDA optimizations
  device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
  args.device = device

  if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    logging.info("CUDA optimizations enabled")

  # Setup logging and create directories
  log_directory: str = setup_logging_and_directories(args)

  # Set random seed for reproducibility
  set_random_seed(args.seed)
  logging.info(f"Random seed set to: {args.seed}")

  # Initialize models and preprocessing transforms
  student_model, preprocess_train, preprocess_val, teacher_model = initialize_models(args, device)

  # Initialize CLIPSelf training method
  clipself_method = CLIPSelf()
  logging.info("CLIPSelf method initialized")

  # Load and prepare training data
  dataloaders: Dict[str, DataInfo] = get_data(args, (preprocess_train, preprocess_val))
  if "train" not in dataloaders:
    raise ValueError("Training data not provided - check data configuration")

  logging.info("Training data loaded successfully")

  # Setup training components (optimizer, scaler)
  optimizer, lr_scheduler, gradient_scaler = setup_training_components(student_model, args)

  # Handle checkpoint resuming
  starting_epoch: int = load_checkpoint_if_resuming(student_model, optimizer, gradient_scaler, args)

  # Main training loop
  logging.info(f"Starting training for {args.epochs} epochs")
  for current_epoch in range(starting_epoch, args.epochs):
    logging.info(f"=== Epoch {current_epoch + 1}/{args.epochs} ===")

    # Train for one epoch
    train_one_epoch(
      student_model=student_model,
      training_method=clipself_method,
      dataloaders=dataloaders,
      loss_fn=None,  # CLIPSelf handles loss internally
      current_epoch=current_epoch,
      optimizer=optimizer,
      gradient_scaler=gradient_scaler,
      lr_scheduler=lr_scheduler,
      teacher_model=teacher_model,
      args=args,
    )

    # Save checkpoint after each epoch
    save_training_checkpoint(student_model, optimizer, gradient_scaler, current_epoch, log_directory)

  logging.info("Training completed successfully")


if __name__ == "__main__":
  main()
