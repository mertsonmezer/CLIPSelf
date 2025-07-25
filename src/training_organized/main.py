"""
CLIPSelf Training Main Entry Point

This is a complete implementation that replicates the functionality of the original
src/training/main.py but uses the reorganized structure.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

# CLIP model creation
from open_clip import create_model, create_model_and_transforms, get_tokenizer

# Original training components (we reuse these for compatibility)
from training.data import get_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.file_utils import pt_load
from training.params import parse_args
from training.scheduler import const_lr, const_lr_cooldown, cosine_lr
from training.train import evaluate, student_teacher_ensemble

# Our reorganized components
from training_organized.utils.logging import setup_logging

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed: int = 42, rank: int = 0):
  """Set random seed for reproducibility."""
  torch.manual_seed(seed + rank)
  np.random.seed(seed + rank)
  random.seed(seed + rank)


def natural_key(string_: str) -> List[int | str]:
  """Natural sorting key for filenames."""
  return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def get_latest_checkpoint(path: str, remote: bool = False):
  """Get the latest checkpoint from a directory."""
  if remote:
    result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 1:
      return None
    checkpoints = [os.path.join(path, x.split(" ")[-1]) for x in result.stdout.decode().split("\n")[:-1]]
  else:
    checkpoints = glob.glob(path + "**/*.pt", recursive=True)
  if checkpoints:
    checkpoints = sorted(checkpoints, key=natural_key)
    return checkpoints[-1]
  return None


def setup_experiment(args):
  """Setup experiment directory and logging."""
  # Create experiment name if not provided
  if args.name is None:
    # Sanitize model name for filesystem use
    model_name_safe = args.model.replace("/", "-")
    date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if args.distributed:
      # Sync date_str from master to all ranks
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

  # Setup experiment directory
  log_base_path = os.path.join(args.logs, args.name)
  args.log_path = None
  if is_master(args, local=args.log_local):
    os.makedirs(log_base_path, exist_ok=True)
    log_filename: str = f"out-{args.rank}" if args.log_local else "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)
    if os.path.exists(args.log_path):
      print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
      return -1

  # Setup logging
  args.log_level = logging.DEBUG if args.debug else logging.INFO
  setup_logging(args.log_path, args.log_level)

  # Set checkpoint path
  args.checkpoint_path = os.path.join(log_base_path, "checkpoints")

  logging.info(f"Experiment directory: {log_base_path}")
  return log_base_path


def create_models_and_transforms(args, device):
  """Create student and teacher models with transforms."""
  logging.info("Creating models and transforms...")

  # Force image size handling
  if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
    args.force_image_size = args.force_image_size[0]

  # Create main model with transforms
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

  args.input_size = model.visual.image_size

  # Create teacher model for distillation (unless using region_clip)
  dist_model = None
  if args.dataset_type == "region_clip":
    logging.info(f"{args.dataset_type}, set dist_model as None")
    dist_model = None
  else:
    logging.info(f"{args.dataset_type}, use dist_model")
    dist_model = create_model(
      args.model, args.pretrained, device=device, precision=args.precision, output_dict=True, cache_dir=args.cache_dir
    )

  # Apply model configurations
  if args.lock_image:
    model.lock_image_tower(
      unlocked_groups=args.lock_image_unlocked_groups,
      freeze_bn_stats=args.lock_image_freeze_bn_stats,
    )

  if args.grad_checkpointing:
    model.set_grad_checkpointing()

  return model, dist_model, (preprocess_train, preprocess_val)


def create_optimizer_and_scaler(args, model):
  """Create optimizer and gradient scaler."""
  if not args.train_data:
    return None, None

  # Parameter groups for different weight decay
  def exclude_params(n, p):
    return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n

  def include_params(n, p):
    return not exclude_params(n, p)

  named_parameters = list(model.named_parameters())
  gain_or_bias_params = [p for n, p in named_parameters if exclude_params(n, p) and p.requires_grad]
  rest_params = [p for n, p in named_parameters if include_params(n, p) and p.requires_grad]

  optimizer = optim.AdamW(
    [
      {"params": gain_or_bias_params, "weight_decay": 0.0},
      {"params": rest_params, "weight_decay": args.wd},
    ],
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=args.eps,
  )

  scaler = GradScaler() if args.precision == "amp" else None
  return optimizer, scaler


def load_checkpoint_if_needed(args, model, optimizer, scaler):
  """Load checkpoint if resuming training."""
  start_epoch = 0
  if args.resume is not None:
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
      # Resuming a train checkpoint with epoch and optimizer state
      start_epoch = checkpoint["epoch"]
      sd = checkpoint["state_dict"]
      if not args.distributed and next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
      model.load_state_dict(sd)
      if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
      if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
      logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
    else:
      # Loading a bare (model only) checkpoint for fine-tune or evaluation
      model.load_state_dict(checkpoint)
      logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
  return start_epoch


def create_scheduler(args, optimizer, data):
  """Create learning rate scheduler."""
  if "train" not in data or optimizer is None:
    return None

  total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs

  if args.lr_scheduler == "cosine":
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
  elif args.lr_scheduler == "const":
    scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
  elif args.lr_scheduler == "const-cooldown":
    assert args.epochs_cooldown is not None, "Please specify the number of cooldown epochs for this lr schedule."
    cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
    scheduler = const_lr_cooldown(
      optimizer, args.lr, args.warmup, total_steps, cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end
    )
  else:
    logging.error(f"Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.")
    exit(1)

  return scheduler


def setup_distributed_training(args, model, dist_model, method):
  """Setup distributed training if needed."""
  if args.distributed:
    if args.use_bn_sync:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_args = {}
    if args.ddp_static_graph:
      ddp_args["static_graph"] = True

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device], **ddp_args)

    if args.dataset_type == "region_clip":
      method = torch.nn.parallel.DistributedDataParallel(method, device_ids=[args.device], **ddp_args)
    if dist_model is not None:
      dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[args.device], **ddp_args)

  return model, dist_model, method


def save_checkpoint(args, epoch, model, optimizer, scaler, dist_model=None):
  """Save training checkpoint."""
  if not is_master(args):
    return

  # Get model state dict (handle ensemble if needed)
  student_state_dict = model.module.state_dict() if args.distributed else model.state_dict()

  if args.alpha < 1.0 and dist_model is not None:
    teacher_state_dict = dist_model.module.state_dict() if args.distributed else dist_model.state_dict()
    target_state_dict = student_teacher_ensemble(student_state_dict, teacher_state_dict, args.alpha)
  else:
    target_state_dict = student_state_dict

  checkpoint_dict = {
    "epoch": epoch,
    "name": args.name,
    "state_dict": target_state_dict,
    "optimizer": optimizer.state_dict(),
  }

  if scaler is not None:
    checkpoint_dict["scaler"] = scaler.state_dict()

  # Save regular checkpoint
  if epoch == args.epochs or (args.save_frequency > 0 and (epoch % args.save_frequency) == 0):
    torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{epoch}.pt"))

  # Delete previous checkpoint if requested
  if args.delete_previous_checkpoint:
    previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{epoch - 1}.pt")
    if os.path.exists(previous_checkpoint):
      os.remove(previous_checkpoint)

  # Save most recent checkpoint
  if args.save_most_recent:
    tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
    latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
    torch.save(checkpoint_dict, tmp_save_path)
    os.replace(tmp_save_path, latest_save_path)


def run_evaluation(args, model, data, epoch, dist_model=None):
  """Run evaluation with the current model."""
  if epoch % args.zeroshot_frequency != 0:
    return

  # Get target state dict (with ensemble if needed)
  student_state_dict = model.module.state_dict() if args.distributed else model.state_dict()

  if args.alpha < 1.0 and dist_model is not None:
    teacher_state_dict = dist_model.module.state_dict() if args.distributed else dist_model.state_dict()
    target_state_dict = student_teacher_ensemble(student_state_dict, teacher_state_dict, args.alpha)
  else:
    target_state_dict = student_state_dict

  # Create test model
  test_model = create_model(
    args.model,
    args.pretrained,
    device=args.device,
    precision=args.precision,
    output_dict=True,
    cache_dir=args.cache_dir,
  )
  test_model.load_state_dict(target_state_dict)

  if args.distributed:
    ddp_args = {}
    if args.ddp_static_graph:
      ddp_args["static_graph"] = True
    test_model = torch.nn.parallel.DistributedDataParallel(test_model, device_ids=[args.device], **ddp_args)

  # Run evaluation
  evaluate(test_model, data, epoch, args)
  del test_model


def main(args_list=None):
  """Main training function that replicates the original main.py functionality."""
  # Parse arguments using the original parser
  args = parse_args(args_list if args_list is not None else sys.argv[1:])

  # CUDA optimizations
  if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

  # Initialize distributed training
  device = init_distributed_device(args)

  # Setup experiment and logging
  exp_dir = setup_experiment(args)
  if exp_dir == -1:
    return -1

  # Log configuration
  if args.precision == "fp16":
    logging.warning(
      "It is recommended to use AMP mixed-precision instead of FP16. "
      "FP16 support needs further verification and tuning, especially for train."
    )

  if args.distributed:
    logging.info(
      f"Running in distributed mode with multiple processes. Device: {args.device}."
      f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
    )
  else:
    logging.info(f"Running with a single process. Device {args.device}.")

  # Set random seed
  random_seed(args.seed, 0)

  # Create models and transforms
  model, dist_model, preprocess_fns = create_models_and_transforms(args, device)

  # Create CLIPSelf method
  from training.clipself import CLIPSelf
  from training.region_clip import RegionCLIP

  if args.dataset_type in ["grid_distill", "proposals_distill"]:
    method = CLIPSelf()
  elif args.dataset_type == "region_clip":
    method = RegionCLIP(args=args).to(device)
  else:
    raise NotImplementedError(f"Dataset type {args.dataset_type} not implemented")

  # Set random seed again after model creation
  random_seed(args.seed, args.rank)

  # Log model info
  if is_master(args):
    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    params_file = os.path.join(args.logs, args.name, "params.txt")
    with open(params_file, "w") as f:
      for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")
        f.write(f"{name}: {val}\n")

  # Setup distributed training
  model, dist_model, method = setup_distributed_training(args, model, dist_model, method)

  # Create optimizer and scaler
  optimizer, scaler = create_optimizer_and_scaler(args, model)

  # Load checkpoint if resuming
  start_epoch = load_checkpoint_if_needed(args, model, optimizer, scaler)

  # Initialize datasets
  data = get_data(args, preprocess_fns, epoch=start_epoch, tokenizer=get_tokenizer(args.model))
  assert len(data), "At least one train or eval dataset must be specified."

  # Create scheduler
  scheduler = create_scheduler(args, optimizer, data)
  if scheduler is None and args.train_data and optimizer is not None:
    return -1  # Error in scheduler creation

  # Determine if this worker should save logs and checkpoints
  args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)

  # Create checkpoint directory
  os.makedirs(args.checkpoint_path, exist_ok=True)

  # Evaluate before training
  if "train" not in data:
    del dist_model
    evaluate(model, data, start_epoch, args)
    return

  logging.info("Evaluate before training")
  evaluate(model, data, start_epoch, args)

  # Training loop
  from training.train import train_one_epoch

  for epoch in range(start_epoch, args.epochs):
    if is_master(args):
      logging.info(f"Start epoch {epoch}")

    # Train one epoch
    train_one_epoch(model, method, data, None, epoch, optimizer, scaler, scheduler, dist_model, args)

    completed_epoch = epoch + 1

    # Save checkpoint
    save_checkpoint(args, completed_epoch, model, optimizer, scaler, dist_model)

    # Run evaluation
    run_evaluation(args, model, data, completed_epoch, dist_model)

  logging.info("Training completed successfully!")


if __name__ == "__main__":
  main()
