"""
CLIPSelf Training Parameters

This module defines all command-line arguments for training CLIPSelf models.
Parameters are organized into logical groups for better understanding and maintenance.
"""

import argparse
import ast
from typing import Dict, Any, Optional, List, Sequence


def get_default_params(model_name: str) -> Dict[str, Any]:
  """
  Get default optimizer parameters based on model architecture.

  Args:
      model_name (str): Name of the model architecture

  Returns:
      Dict[str, Any]: Default learning rate, beta1, beta2, and epsilon values

  Note:
      Parameters are based on the original CLIP paper (https://arxiv.org/pdf/2103.00020.pdf)
      ViT models use different optimizer settings than ResNet models.
  """
  model_name = model_name.lower()
  if "vit" in model_name:
    return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
  else:
    return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
  """
  Custom argparse action to parse keyword arguments from command line.

  Allows passing dictionary-like arguments in the format: key1=value1 key2=value2
  Automatically tries to evaluate values as Python literals, falls back to strings.
  """

  def __call__(
    self,
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    values: Optional[Sequence[Any]],
    option_string: Optional[str] = None,
  ) -> None:
    kw = {}
    if values:
      for value in values:
        key, val = value.split("=")
        try:
          kw[key] = ast.literal_eval(val)
        except ValueError:
          kw[key] = str(val)  # fallback to string (avoid need to escape on command line)
    setattr(namespace, self.dest, kw)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
  """
  Parse command-line arguments for CLIPSelf training.

  Args:
      args: Command-line arguments (typically sys.argv[1:])

  Returns:
      argparse.Namespace: Parsed arguments with default values applied
  """
  parser = argparse.ArgumentParser(
    description="Train CLIPSelf models with different dataset types and configurations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  # ============================================================================
  # DATASET CONFIGURATION
  # ============================================================================
  dataset_group = parser.add_argument_group(
    "Dataset Configuration", "Settings for data loading, preprocessing, and dataset selection"
  )

  dataset_group.add_argument(
    "--dataset-type",
    choices=["proposals_distill", "region_clip", "grid_distill"],
    default="grid_distill",
    help="Training method: 'grid_distill' for patch-based learning, "
    "'proposals_distill' for object-based learning, 'region_clip' for region-text alignment",
  )

  dataset_group.add_argument(
    "--test-type",
    choices=["coco_panoptic"],
    default="coco_panoptic",
    help="Evaluation dataset type for zero-shot testing",
  )

  dataset_group.add_argument(
    "--train-data",
    type=str,
    default="",
    help="Path to training annotation file (COCO-style JSON). "
    "Use instances_train2017.json for grid_distill, coco_proposals.json for proposals_distill",
  )

  dataset_group.add_argument(
    "--val-data",
    type=str,
    default="data/coco/annotations/instances_val2017_100.json",
    help="Path to validation annotation file for evaluation",
  )

  dataset_group.add_argument(
    "--train-image-root",
    type=str,
    default="data/coco/val2017",
    help="Root directory containing training images (e.g., data/coco/train2017)",
  )

  dataset_group.add_argument(
    "--val-image-root", type=str, default="data/coco/val2017", help="Root directory containing validation images"
  )

  dataset_group.add_argument(
    "--train-ceph-root",
    type=str,
    default="",
    help="Ceph distributed storage root path for training images (optional, leave empty for local storage)",
  )

  dataset_group.add_argument(
    "--train-ratio",
    type=float,
    default=1.0,
    help="Fraction of training data to use (0.0-1.0). Useful for quick experiments with subset of data",
  )

  # ============================================================================
  # OBJECT DETECTION & SEGMENTATION SETTINGS
  # ============================================================================
  detection_group = parser.add_argument_group(
    "Object Detection & Segmentation", "Parameters for object proposals, bounding boxes, and segmentation masks"
  )

  detection_group.add_argument(
    "--max-boxes",
    type=int,
    default=20,
    help="Maximum number of object proposals/annotations per image. "
    "Higher values capture more objects but increase memory usage",
  )

  detection_group.add_argument(
    "--max-masks",
    type=int,
    default=20,
    help="Maximum number of segmentation masks per image (for panoptic segmentation)",
  )

  detection_group.add_argument(
    "--min-size",
    type=int,
    default=8,
    help="Minimum object size in pixels (width x height). Objects smaller than this are filtered out",
  )

  detection_group.add_argument(
    "--max-size",
    type=int,
    default=1024,
    help="Maximum object size in pixels (width x height). Objects larger than this are filtered out",
  )

  detection_group.add_argument(
    "--crop-scale",
    type=float,
    default=1.0,
    help="Scale factor for expanding crop regions around objects. "
    "1.0 = no expansion, 1.5 = expand by 50% for more context",
  )

  detection_group.add_argument(
    "--box-scale", type=float, default=1.5, help="Scale factor for expanding bounding boxes in proposal generation"
  )

  detection_group.add_argument(
    "--mask-thr", type=float, default=0.7, help="Threshold for mask confidence scores (0.0-1.0)"
  )

  # ============================================================================
  # GRID DISTILLATION SETTINGS
  # ============================================================================
  grid_group = parser.add_argument_group(
    "Grid Distillation", "Parameters specific to grid-based patch learning (dataset-type=grid_distill)"
  )

  grid_group.add_argument(
    "--max-split",
    type=int,
    default=6,
    help="Maximum grid size for patch generation. Creates grids from 1x1 up to max-splitxmax-split. "
    "Higher values generate more fine-grained patches but increase computation",
  )

  grid_group.add_argument(
    "--grid-noise",
    action="store_true",
    default=False,
    help="Add random noise to grid coordinates for data augmentation",
  )

  grid_group.add_argument(
    "--shift-range",
    type=float,
    default=0.0,
    help="Random shift range for grid coordinates (as fraction of image size)",
  )

  grid_group.add_argument(
    "--scale-range", type=float, default=0.0, help="Random scale variation range for grid patches"
  )

  # ============================================================================
  # IMAGE PROCESSING & TRANSFORMS
  # ============================================================================
  image_group = parser.add_argument_group("Image Processing", "Image preprocessing, augmentation, and size settings")

  image_group.add_argument(
    "--det-image-size",
    type=int,
    default=1024,
    help="Image size for object detection preprocessing. Larger sizes capture more detail but require more memory",
  )

  image_group.add_argument(
    "--train-image-size", type=int, default=1024, help="Base image size for training preprocessing"
  )

  image_group.add_argument(
    "--force-image-size",
    type=int,
    nargs="+",
    default=None,
    help="Override default image size from model configuration. "
    "Provide one value for square images or two values for [height, width]",
  )

  image_group.add_argument(
    "--image-mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override default image normalization mean values (RGB order)",
  )

  image_group.add_argument(
    "--image-std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override default image normalization standard deviation values (RGB order)",
  )

  image_group.add_argument(
    "--pre-transforms",
    action="store_true",
    default=False,
    help="Apply additional data augmentation transforms (random resize, crop, horizontal flip)",
  )

  image_group.add_argument(
    "--multiscale", action="store_true", default=False, help="Enable multi-scale training with varying image sizes"
  )

  image_group.add_argument(
    "--aug-cfg",
    nargs="*",
    default={},
    action=ParseKwargs,
    help="Additional augmentation configuration as key=value pairs",
  )

  # ============================================================================
  # MODEL ARCHITECTURE
  # ============================================================================
  model_group = parser.add_argument_group(
    "Model Architecture", "Model selection, architecture settings, and feature dimensions"
  )

  model_group.add_argument(
    "--model",
    type=str,
    default="RN50",
    help="Vision backbone architecture. Examples: RN50, RN101, ViT-B/32, ViT-L/14, EVA02-CLIP-B-16",
  )

  model_group.add_argument(
    "--pretrained",
    default="",
    type=str,
    help="Pretrained model weights. Use 'openai' for OpenAI CLIP weights, "
    "'eva' for EVA-CLIP weights, or path to custom checkpoint",
  )

  model_group.add_argument(
    "--pretrained-image",
    default=False,
    action="store_true",
    help="Load ImageNet pretrained weights for image tower backbone (if available)",
  )

  model_group.add_argument(
    "--embed-dim", type=int, default=768, help="Embedding dimension for image and text features"
  )

  model_group.add_argument(
    "--extract-type",
    type=str,
    choices=["v1", "v2"],
    default="v2",
    help="Feature extraction method version. v2 is recommended for better performance",
  )

  model_group.add_argument(
    "--force-quick-gelu",
    default=False,
    action="store_true",
    help="Force use of QuickGELU activation for non-OpenAI transformer models",
  )

  model_group.add_argument(
    "--force-custom-text",
    default=False,
    action="store_true",
    help="Force use of CustomTextCLIP model with separate text tower",
  )

  model_group.add_argument(
    "--force-patch-dropout",
    default=None,
    type=float,
    help="Override patch dropout rate during training. "
    "Useful for fine-tuning with reduced dropout near the end of training",
  )

  # ============================================================================
  # MODEL FREEZING & FINE-TUNING
  # ============================================================================
  freeze_group = parser.add_argument_group(
    "Model Freezing & Fine-tuning", "Control which parts of the model are trainable"
  )

  freeze_group.add_argument(
    "--lock-image",
    default=False,
    action="store_true",
    help="Freeze the image tower by disabling gradients. Useful for fine-tuning only specific components",
  )

  freeze_group.add_argument(
    "--lock-image-unlocked-groups",
    type=int,
    default=3,
    help="Number of image tower layer groups to keep unfrozen when --lock-image is used. "
    "Allows fine-tuning of top layers while freezing lower layers",
  )

  freeze_group.add_argument(
    "--lock-image-freeze-bn-stats",
    default=True,
    action="store_true",
    help="Freeze BatchNorm running statistics in locked image tower layers",
  )

  # ============================================================================
  # TRAINING HYPERPARAMETERS
  # ============================================================================
  training_group = parser.add_argument_group(
    "Training Hyperparameters", "Learning rate, optimization, and training schedule settings"
  )

  training_group.add_argument("--epochs", type=int, default=32, help="Number of training epochs")

  training_group.add_argument(
    "--batch-size", type=int, default=64, help="Batch size per GPU. Total batch size = batch_size × num_gpus"
  )

  training_group.add_argument(
    "--lr",
    type=float,
    default=1e-5,
    help="Learning rate. Will be scaled by batch size and number of GPUs in distributed training",
  )

  training_group.add_argument(
    "--beta1", type=float, default=None, help="Adam optimizer beta1 parameter. If None, uses model-specific default"
  )

  training_group.add_argument(
    "--beta2", type=float, default=None, help="Adam optimizer beta2 parameter. If None, uses model-specific default"
  )

  training_group.add_argument(
    "--eps", type=float, default=None, help="Adam optimizer epsilon parameter. If None, uses model-specific default"
  )

  training_group.add_argument("--wd", type=float, default=0.2, help="Weight decay (L2 regularization) strength")

  training_group.add_argument(
    "--warmup", type=int, default=10000, help="Number of warmup steps for learning rate schedule"
  )

  training_group.add_argument(
    "--lr-scheduler",
    type=str,
    default="cosine",
    help="Learning rate scheduler type: 'cosine', 'const' (constant), or 'const-cooldown'",
  )

  training_group.add_argument(
    "--lr-cooldown-end", type=float, default=0.0, help="Final learning rate for cooldown schedule"
  )

  training_group.add_argument(
    "--lr-cooldown-power", type=float, default=1.0, help="Power for polynomial cooldown schedule (1.0 = linear decay)"
  )

  training_group.add_argument(
    "--skip-scheduler",
    action="store_true",
    default=False,
    help="Skip learning rate decay and use constant learning rate",
  )

  training_group.add_argument(
    "--grad-clip-norm",
    type=float,
    default=None,
    help="Gradient clipping norm. Clips gradients if their norm exceeds this value",
  )

  training_group.add_argument(
    "--accum-freq", type=int, default=1, help="Gradient accumulation frequency. Updates model every N steps"
  )

  # ============================================================================
  # LOSS FUNCTION WEIGHTS
  # ============================================================================
  loss_group = parser.add_argument_group(
    "Loss Function Configuration", "Weights for different loss components in CLIPSelf training"
  )

  loss_group.add_argument(
    "--alpha", type=float, default=2.0, help="Alpha parameter for loss weighting. Not used when alpha >= 1.0"
  )

  loss_group.add_argument("--kl-weight", type=float, default=1.0, help="Weight for KL divergence loss in distillation")

  loss_group.add_argument("--contrast-weight", type=float, default=1.0, help="Weight for contrastive loss component")

  loss_group.add_argument("--l1-weight", type=float, default=0.10, help="Weight for L1 regularization loss")

  loss_group.add_argument("--smooth-weight", type=float, default=0.0, help="Weight for smoothness regularization")

  loss_group.add_argument("--cosine-weight", type=float, default=1.0, help="Weight for cosine similarity loss")

  loss_group.add_argument(
    "--fix-logit-scale",
    action="store_true",
    default=False,
    help="Fix the logit scale parameter instead of learning it",
  )

  # ============================================================================
  # SEGMENTATION & EMBEDDINGS
  # ============================================================================
  segmentation_group = parser.add_argument_group(
    "Segmentation & Text Embeddings", "Settings for panoptic segmentation and precomputed text embeddings"
  )

  segmentation_group.add_argument(
    "--val-segm-root",
    type=str,
    default="data/coco/annotations/panoptic_val2017",
    help="Directory containing validation segmentation masks",
  )

  segmentation_group.add_argument(
    "--train-segm-root",
    type=str,
    default="data/coco/annotations/panoptic_val2017",
    help="Directory containing training segmentation masks",
  )

  segmentation_group.add_argument(
    "--downsample-factor",
    type=int,
    default=16,
    help="Downsampling factor for segmentation masks to reduce memory usage",
  )

  segmentation_group.add_argument(
    "--embed-path",
    type=str,
    default="metadata/coco_clip_hand_craft_RN50.npy",
    help="Path to precomputed text embeddings for class names (.npy file)",
  )

  segmentation_group.add_argument(
    "--train-embed-path", type=str, default="", help="Path to precomputed text embeddings for training (optional)"
  )

  # ============================================================================
  # COMPUTATION & MEMORY
  # ============================================================================
  computation_group = parser.add_argument_group(
    "Computation & Memory", "Settings for memory usage, precision, and computational efficiency"
  )

  computation_group.add_argument(
    "--precision",
    choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
    default="amp",
    help="Floating point precision: 'amp' (automatic mixed precision), 'fp16', 'bf16', or 'fp32'",
  )

  computation_group.add_argument(
    "--grad-checkpointing",
    default=False,
    action="store_true",
    help="Enable gradient checkpointing to reduce memory usage at cost of computation",
  )

  computation_group.add_argument(
    "--torchscript",
    default=False,
    action="store_true",
    help="Compile model with TorchScript for potential performance gains",
  )

  computation_group.add_argument(
    "--workers", type=int, default=1, help="Number of data loader worker processes per GPU"
  )

  computation_group.add_argument(
    "--use-bn-sync", default=False, action="store_true", help="Use synchronized batch normalization across GPUs"
  )

  computation_group.add_argument(
    "--gather-with-grad",
    default=False,
    action="store_true",
    help="Enable full distributed gradient for feature gathering (increases memory usage)",
  )

  # ============================================================================
  # DISTRIBUTED TRAINING
  # ============================================================================
  distributed_group = parser.add_argument_group(
    "Distributed Training", "Multi-GPU and multi-node training configuration"
  )

  distributed_group.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="URL for distributed training setup. 'env://' reads from environment variables",
  )

  distributed_group.add_argument(
    "--dist-backend", default="nccl", type=str, help="Distributed training backend: 'nccl' for GPUs, 'gloo' for CPUs"
  )

  distributed_group.add_argument(
    "--horovod", default=False, action="store_true", help="Use Horovod instead of PyTorch native distributed training"
  )

  distributed_group.add_argument(
    "--ddp-static-graph",
    default=False,
    action="store_true",
    help="Enable static graph optimization for DistributedDataParallel (PyTorch >= 1.11)",
  )

  distributed_group.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't automatically set device from local rank (useful when CUDA_VISIBLE_DEVICES is set)",
  )

  # ============================================================================
  # CHECKPOINTING & EVALUATION
  # ============================================================================
  checkpoint_group = parser.add_argument_group(
    "Checkpointing & Evaluation", "Model saving, loading, and evaluation frequency settings"
  )

  checkpoint_group.add_argument(
    "--cache-dir",
    type=str,
    default="checkpoints",
    help="Directory to cache/store model checkpoints and pretrained weights",
  )

  checkpoint_group.add_argument("--save-frequency", type=int, default=1, help="Save checkpoint every N epochs")

  checkpoint_group.add_argument(
    "--save-most-recent",
    action="store_true",
    default=False,
    help="Always save the most recent model as 'epoch_latest.pt'",
  )

  checkpoint_group.add_argument(
    "--delete-previous-checkpoint",
    default=False,
    action="store_true",
    help="Delete previous checkpoint when saving a new one (saves disk space)",
  )

  checkpoint_group.add_argument(
    "--resume", default=None, type=str, help="Path to checkpoint file to resume training from"
  )

  checkpoint_group.add_argument(
    "--zeroshot-frequency", type=int, default=2, help="Run zero-shot evaluation every N epochs"
  )

  # ============================================================================
  # LOGGING & DEBUGGING
  # ============================================================================
  logging_group = parser.add_argument_group(
    "Logging & Debugging", "Experiment tracking, logging, and debugging options"
  )

  logging_group.add_argument(
    "--logs", type=str, default="./logs/", help="Directory to store TensorBoard logs and training outputs"
  )

  logging_group.add_argument(
    "--name", type=str, default=None, help="Experiment name for logging. If None, uses current timestamp"
  )

  logging_group.add_argument(
    "--log-local",
    action="store_true",
    default=False,
    help="Log files on local master process only (not global master in multi-node setup)",
  )

  logging_group.add_argument("--log-every-n-steps", type=int, default=100, help="Log training metrics every N steps")

  logging_group.add_argument(
    "--debug", default=False, action="store_true", help="Enable debug mode with additional logging and checks"
  )

  logging_group.add_argument(
    "--copy-codebase",
    default=False,
    action="store_true",
    help="Copy entire codebase to log directory for reproducibility",
  )

  logging_group.add_argument("--seed", type=int, default=0, help="Random seed for reproducible results")

  # ============================================================================
  # ADVANCED OPTIONS
  # ============================================================================
  advanced_group = parser.add_argument_group(
    "Advanced Options", "Advanced settings for specific use cases and experiments"
  )

  advanced_group.add_argument(
    "--image-ave-pool",
    action="store_true",
    default=False,
    help="Use average pooling for image features (experimental)",
  )

  advanced_group.add_argument(
    "--roi-teacher", action="store_true", default=False, help="Use ROI-based teacher model (experimental)"
  )

  advanced_group.add_argument(
    "--del-dist-model",
    action="store_true",
    default=False,
    help="Delete distributed model after training (saves memory)",
  )

  # Parse arguments and apply defaults
  parsed_args = parser.parse_args(args)

  # Apply model-specific default parameters if not explicitly set
  default_params = get_default_params(parsed_args.model)
  for name, val in default_params.items():
    if getattr(parsed_args, name) is None:
      setattr(parsed_args, name, val)

  return parsed_args
