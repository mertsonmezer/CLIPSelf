"""
CLIPSelf Training Main Script

This module provides the main training loop for the CLIPSelf model, supporting both
grid distillation and region clip training methods. It handles distributed training,
checkpointing, evaluation, and student-teacher ensembling.
"""

import argparse
import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import List, Union, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from open_clip import create_model, create_model_and_transforms, get_tokenizer
from training.clipself import CLIPSelf
from training.data import get_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.file_utils import pt_load
from training.logger import setup_logging
from training.params import parse_args
from training.region_clip import RegionCLIP
from training.scheduler import const_lr, const_lr_cooldown, cosine_lr
from training.train import evaluate, student_teacher_ensemble, train_one_epoch

# Constant for the latest checkpoint filename
LATEST_CHECKPOINT_NAME: str = "epoch_latest.pt"


def random_seed(seed: int = 42, rank: int = 0) -> None:
  """
  Set random seeds for reproducible results across different frameworks.

  Args:
    seed (int): Base random seed value
    rank (int): Process rank for distributed training (added to seed for variation)
  """
  torch.manual_seed(seed + rank)
  np.random.seed(seed + rank)
  random.seed(seed + rank)


def natural_key(string_: str) -> List[Union[int, str]]:
  """
  Natural sorting key function for alphanumeric strings.

  Enables proper sorting of strings containing numbers, e.g.,
  ['epoch_1.pt', 'epoch_2.pt', 'epoch_10.pt'] instead of
  ['epoch_1.pt', 'epoch_10.pt', 'epoch_2.pt']

  See: http://www.codinghorror.com/blog/archives/001018.html

  Args:
    string_ (str): Input string to create sorting key for

  Returns:
    List of integers and strings for natural sorting
  """
  return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool) -> Optional[str]:
  """
  Find the most recent checkpoint file in the given path.

  Args:
    path (str): Directory path to search for checkpoints
    remote (bool): Whether to search in AWS S3 (True) or local filesystem (False)

  Returns:
    Path to the latest checkpoint file, or None if no checkpoints found
  """
  # Search for checkpoints recursively, can pick up checkpoints across multiple sub-folders
  if remote:
    result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result)
    if result.returncode == 1:
      return None
    checkpoints: List[str] = [os.path.join(path, x.split(" ")[-1]) for x in result.stdout.decode().split("\n")[:-1]]
  else:
    checkpoints: List[str] = glob.glob(path + "**/*.pt", recursive=True)

  if checkpoints:
    checkpoints = sorted(checkpoints, key=natural_key)
    return checkpoints[-1]
  return None


def main(command_line_args: List[str]) -> None:
  """
  Main training function for CLIPSelf model.

  This function orchestrates the entire training pipeline including:
  - Argument parsing and validation
  - Device initialization and distributed setup
  - Model creation and configuration
  - Dataset loading and preprocessing
  - Training loop with checkpointing
  - Evaluation and logging

  Args:
    command_line_args: List of command line arguments
  """
  # Parse command line arguments
  args: argparse.Namespace = parse_args(command_line_args)

  # Configure CUDA optimizations if available
  if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

  # Initialize device and distributed training
  device: torch.device = init_distributed_device(args)

  # Generate experiment name if not provided
  if args.name is None:
    model_name_safe: str = args.model.replace("/", "-")
    date_str: str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
      # Synchronize date string from master to all ranks for consistency
      date_str = broadcast_object(args, date_str)
    args.name = "-".join(
      [
        date_str,
        f"model_{model_name_safe}",
        f"lr_{args.lr}",
        f"b_{args.batch_size}",
        f"j_{args.workers}",
        f"p_{args.precision}",
      ]
    )

  # Setup logging paths and directories
  log_base_path: str = os.path.join(args.logs, args.name)
  args.log_path = None
  if is_master(args, local=args.log_local):
    os.makedirs(log_base_path, exist_ok=True)
    log_filename: str = f"out-{args.rank}" if args.log_local else "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)
    if os.path.exists(args.log_path):
      print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
      return

  # Configure logging
  args.log_level = logging.DEBUG if args.debug else logging.INFO
  setup_logging(args.log_path, args.log_level)
  args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

  # Warn about FP16 precision
  if args.precision == "fp16":
    logging.warning(
      "It is recommended to use AMP mixed-precision instead of FP16. "
      "FP16 support needs further verification and tuning, especially for train."
    )

  # Log distributed training information
  if args.distributed:
    logging.info(
      f"Running in distributed mode with multiple processes. Device: {args.device}. "
      f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
    )
  else:
    logging.info(f"Running with a single process. Device {args.device}.")

  # Handle force_image_size argument conversion
  if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
    # Convert single-element list/tuple to integer (for square images)
    args.force_image_size = args.force_image_size[0]

  # Set random seed for reproducibility
  random_seed(args.seed, 0)

  # Create model and data transforms
  model, preprocess_train, preprocess_val = create_model_and_transforms(
    args.model,
    args.pretrained,
    precision=args.precision,
    device=device,
    jit=args.torchscript,
    force_quick_gelu=args.force_quick_gelu,
    force_custom_text=args.force_custom_text,
    force_patch_dropout=args.force_patch_dropout,
    force_image_size=args.force_image_size,
    pretrained_image=args.pretrained_image,
    image_mean=args.image_mean,
    image_std=args.image_std,
    aug_cfg=args.aug_cfg,
    output_dict=True,
    cache_dir=args.cache_dir,
    det_image_size=args.det_image_size,
    dataset_type=args.dataset_type,
  )

  # Store input size from model for later use
  args.input_size = model.visual.image_size

  # Initialize training method based on dataset type
  method: Union[CLIPSelf, RegionCLIP]
  if args.dataset_type in ["grid_distill", "proposals_distill"]:
    method = CLIPSelf()
  elif args.dataset_type == "region_clip":
    method = RegionCLIP(args=args).to(device)
  else:
    raise NotImplementedError(f"Unknown dataset_type: {args.dataset_type}")

  # Initialize distillation model (teacher model) if needed
  dist_model: Optional[Any] = None
  if args.dataset_type == "region_clip":
    logging.info(f"{args.dataset_type}, set dist_model as None")
    dist_model = None
  else:
    logging.info(f"{args.dataset_type}, use dist_model")
    dist_model = create_model(
      args.model,  # Use same model architecture as student
      args.pretrained,
      device=device,
      precision=args.precision,
      output_dict=True,
      cache_dir=args.cache_dir,  # Cache directory for pre-trained models
    )

  # Re-seed with rank for distributed training variance
  random_seed(args.seed, args.rank)

  # Configure model training options
  if args.lock_image:
    # Lock image tower as per LiT (https://arxiv.org/abs/2111.07991)
    model.lock_image_tower(
      unlocked_groups=args.lock_image_unlocked_groups,
      freeze_bn_stats=args.lock_image_freeze_bn_stats,
    )
  if args.grad_checkpointing:
    model.set_grad_checkpointing()

  # Log model information (master process only)
  if is_master(args):
    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    params_file: str = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
      for name in sorted(vars(args)):
        val: Any = getattr(args, name)
        logging.info(f"  {name}: {val}")
        f.write(f"{name}: {val}\n")

  # Setup distributed data parallel if needed
  if args.distributed:
    if args.use_bn_sync:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_args: Dict[str, Any] = {}  # Additional arguments for DistributedDataParallel
    if args.ddp_static_graph:
      # Static graph optimization (available in newer PyTorch versions)
      ddp_args["static_graph"] = True
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    if args.dataset_type == "region_clip":
      method = torch.nn.parallel.DistributedDataParallel(method, device_ids=[device], **ddp_args)
    if dist_model is not None:
      dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

  # Create optimizer and gradient scaler
  optimizer: Optional[Any] = None
  scaler: Optional[GradScaler] = None

  if args.train_data:
    # Define parameter filtering functions for weight decay
    def should_exclude_from_weight_decay(name: str, param: torch.Tensor) -> bool:
      """Check if parameter should be excluded from weight decay."""
      return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    def should_include_in_weight_decay(name: str, param: torch.Tensor) -> bool:
      """Check if parameter should be included in weight decay."""
      return not should_exclude_from_weight_decay(name, param)

    named_parameters: List[Tuple[str, torch.Tensor]] = list(model.named_parameters())
    gain_or_bias_params: List[torch.Tensor] = [
      p for n, p in named_parameters if should_exclude_from_weight_decay(n, p) and p.requires_grad
    ]
    rest_params: List[torch.Tensor] = [
      p for n, p in named_parameters if should_include_in_weight_decay(n, p) and p.requires_grad
    ]

    # Create AdamW optimizer with different weight decay for different parameter groups
    optimizer = optim.AdamW(
      [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": args.wd},
      ],
      lr=args.lr,
      betas=(args.beta1, args.beta2),
      eps=args.eps,
    )
    # Initialize gradient scaler for automatic mixed precision
    scaler = GradScaler() if args.precision == "amp" else None

  # Resume from checkpoint if specified
  start_epoch: int = 0
  if args.resume is not None:
    checkpoint: Dict[str, Any] = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
      # Resuming a training checkpoint with epoch and optimizer state
      start_epoch = checkpoint["epoch"]
      sd: Dict[str, Any] = checkpoint["state_dict"]
      if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
      model.load_state_dict(sd)
      if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
      if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
      logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
      # Loading a bare (model only) checkpoint for fine-tuning or evaluation
      model.load_state_dict(checkpoint)
      logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

  # Initialize datasets and data loaders
  data: Dict[str, Any] = get_data(
    args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model)
  )
  assert len(data), "At least one train or eval dataset must be specified."

  # Create learning rate scheduler for training
  scheduler: Optional[Any] = None
  if "train" in data and optimizer is not None:
    total_steps: int = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
    if args.lr_scheduler == "cosine":
      scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const":
      scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
    elif args.lr_scheduler == "const-cooldown":
      assert args.epochs_cooldown is not None, "Please specify the number of cooldown epochs for this lr schedule."
      cooldown_steps: int = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
      scheduler = const_lr_cooldown(
        optimizer, args.lr, args.warmup, total_steps, cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end
      )
    else:
      logging.error(f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.")
      exit(1)

  # Determine if this worker should save logs and checkpoints (only master process)
  args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)

  # Run initial evaluation before training
  logging.info("Evaluate before training")
  os.makedirs(args.checkpoint_path, exist_ok=True)
  if "train" not in data:
    # If no training data, just evaluate and exit
    del dist_model
    evaluate(model, data, start_epoch, args)
    return
  evaluate(model, data, start_epoch, args)

  loss: Optional[Any] = None  # Loss will be computed in train_one_epoch

  # Main training loop
  for epoch in range(start_epoch, args.epochs):
    if is_master(args):
      logging.info(f"Start epoch {epoch}")

    # Train for one epoch
    train_one_epoch(model, method, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args)
    completed_epoch: int = epoch + 1

    # Get student model state dict (unwrap from DDP if needed)
    student_state_dict: Dict[str, Any] = model.module.state_dict() if args.distributed else model.state_dict()

    # Apply student-teacher ensemble if alpha < 1.0
    if args.alpha < 1.0:
      if dist_model is not None:
        # Get teacher model state dict (unwrap from DDP if needed)
        teacher_state_dict: Dict[str, Any] = (
          dist_model.module.state_dict() if args.distributed else dist_model.state_dict()
        )
      else:
        # Create fresh teacher model if dist_model was None
        dist_model = create_model(
          args.model,
          args.pretrained,
          device=device,
          precision=args.precision,
          output_dict=True,
          cache_dir=args.cache_dir,
        )
        teacher_state_dict: Dict[str, Any] = dist_model.state_dict()
        dist_model = None
      # Ensemble student and teacher weights
      target_state_dict: Dict[str, Any] = student_teacher_ensemble(student_state_dict, teacher_state_dict, args.alpha)
    else:
      # Use only student weights if alpha == 1.0
      target_state_dict: Dict[str, Any] = student_state_dict

    # Save checkpoints (master process only)
    if is_master(args):
      # Prepare checkpoint dictionary
      checkpoint_dict: Dict[str, Any] = {
        "epoch": completed_epoch,
        "name": args.name,
        "state_dict": target_state_dict,
        "optimizer": optimizer.state_dict(),
      }
      if scaler is not None:
        checkpoint_dict["scaler"] = scaler.state_dict()

      # Save checkpoint at specified intervals or at final epoch
      if completed_epoch == args.epochs or (args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0):
        torch.save(
          checkpoint_dict,
          os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
        )

      # Optionally delete previous checkpoint to save disk space
      if args.delete_previous_checkpoint:
        previous_checkpoint: str = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
        if os.path.exists(previous_checkpoint):
          os.remove(previous_checkpoint)

      # Save most recent checkpoint with atomic write
      if args.save_most_recent:
        # Use temporary file to avoid corrupting latest checkpoint if save fails
        tmp_save_path: str = os.path.join(args.checkpoint_path, "tmp.pt")
        latest_save_path: str = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
        torch.save(checkpoint_dict, tmp_save_path)
        os.replace(tmp_save_path, latest_save_path)

    # Periodic evaluation during training
    if completed_epoch % args.zeroshot_frequency == 0:
      # Create fresh model for evaluation to avoid distributed wrapper issues
      test_model: Any = create_model(
        args.model,
        args.pretrained,
        device=device,
        precision=args.precision,
        output_dict=True,
        cache_dir=args.cache_dir,
      )
      test_model.load_state_dict(target_state_dict)
      if args.distributed:
        test_model = torch.nn.parallel.DistributedDataParallel(test_model, device_ids=[device], **ddp_args)
      evaluate(test_model, data, completed_epoch, args)
      del test_model


if __name__ == "__main__":
  main(sys.argv[1:])
