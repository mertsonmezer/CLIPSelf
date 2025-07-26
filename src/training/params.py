"""
CLIPSelf Training Parameters

This module defines all command-line arguments for training CLIPSelf models.
Parameters control dataset loading, model architecture, training settings, and evaluation.
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

  # Core training configuration
  parser.add_argument(
    "--dataset-type",
    choices=["proposals_distill", "region_clip", "grid_distill"],
    default="grid_distill",
    help="Training method: 'grid_distill' uses regular image patches for self-distillation, "
    "'proposals_distill' uses object proposals from coco_proposals.json for region-based distillation, "
    "'region_clip' trains RegionCLIP with region-text pairs from coco_pseudo_4764.json",
  )

  parser.add_argument(
    "--test-type",
    choices=["coco_panoptic"],
    default="coco_panoptic",
    help="Evaluation dataset type for zero-shot testing",
  )

  parser.add_argument(
    "--train-data",
    type=str,
    default="",
    help="Path to training annotation file. Use: instances_train2017.json for grid_distill, "
    "coco_proposals.json for proposals_distill, coco_pseudo_4764.json for region_clip",
  )

  parser.add_argument(
    "--val-data",
    type=str,
    default="data/coco/annotations/instances_val2017_100.json",
    help="Path to validation annotation file (COCO panoptic format for evaluation)",
  )

  parser.add_argument(
    "--train-image-root",
    type=str,
    default="data/coco/val2017",
    help="Root directory containing training images (e.g., data/coco/train2017)",
  )

  parser.add_argument(
    "--val-image-root", type=str, default="data/coco/val2017", help="Root directory containing validation images"
  )

  parser.add_argument(
    "--train-ceph-root",
    type=str,
    default="",
    help="Ceph distributed storage root path for training images (optional, leave empty for local storage)",
  )

  parser.add_argument(
    "--train-ratio",
    type=float,
    default=1.0,
    help="Fraction of training data to use (0.0-1.0). Useful for quick experiments with subset of data",
  )

  # Object detection and proposals
  parser.add_argument(
    "--max-boxes",
    type=int,
    default=20,
    help="Maximum number of object proposals/annotations per image (used as max_anns in datasets). "
    "Controls memory usage - higher values capture more objects but require more memory",
  )

  parser.add_argument(
    "--max-masks",
    type=int,
    default=20,
    help="Maximum number of segmentation masks per image for panoptic segmentation evaluation",
  )

  parser.add_argument(
    "--min-size",
    type=int,
    default=8,
    help="Minimum object size in pixels (width x height). Objects smaller than this are filtered out from proposals",
  )

  parser.add_argument(
    "--max-size",
    type=int,
    default=1024,
    help="Maximum object size in pixels (width x height). Objects larger than this are filtered out from proposals",
  )

  parser.add_argument(
    "--crop-scale",
    type=float,
    default=1.0,
    help="Scale factor for expanding crop regions around grid patches. 1.0=no expansion, 1.5=expand by 50% for more context",
  )

  parser.add_argument(
    "--box-scale",
    type=float,
    default=1.5,
    help="Scale factor for expanding bounding boxes in proposal generation (used in ProposalDistillDataset for context)",
  )

  parser.add_argument(
    "--mask-thr",
    type=float,
    default=0.7,
    help="Threshold for mask confidence scores (0.0-1.0) in segmentation evaluation",
  )

  # Grid distillation specific
  parser.add_argument(
    "--max-split",
    type=int,
    default=6,
    help="Maximum grid size for patch generation in grid_distill mode. Creates grids from 1x1 up to max-split x max-split. "
    "Higher values generate more fine-grained patches but increase computation",
  )

  parser.add_argument(
    "--grid-noise",
    action="store_true",
    default=False,
    help="Add random noise to grid coordinates for data augmentation in grid_distill mode",
  )

  parser.add_argument(
    "--shift-range",
    type=float,
    default=0.0,
    help="Random shift range for grid coordinates (as fraction of image size) in grid_distill mode",
  )

  parser.add_argument(
    "--scale-range", type=float, default=0.0, help="Random scale variation range for grid patches in grid_distill mode"
  )

  # Image processing
  parser.add_argument(
    "--det-image-size",
    type=int,
    default=1024,
    help="Image size for object detection preprocessing. Used for initial image resizing before feature extraction",
  )

  parser.add_argument(
    "--train-image-size",
    type=int,
    default=1024,
    help="Base image size for training preprocessing (typically same as det-image-size)",
  )

  parser.add_argument(
    "--force-image-size",
    type=int,
    nargs="+",
    default=None,
    help="Override default image size from model configuration. "
    "Provide one value for square images or two values for [height, width]",
  )

  parser.add_argument(
    "--image-mean",
    type=float,
    nargs="+",
    default=None,
    metavar="MEAN",
    help="Override default image normalization mean values (RGB order). Example: --image-mean 0.485 0.456 0.406",
  )

  parser.add_argument(
    "--image-std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override default image normalization standard deviation values (RGB order). "
    "Example: --image-std 0.229 0.224 0.225",
  )

  parser.add_argument(
    "--pre-transforms",
    action="store_true",
    default=False,
    help="Apply additional data augmentation transforms in GridDistillDataset: "
    "random resize (0.5-2.0x), crop, and horizontal flip",
  )

  parser.add_argument(
    "--multiscale",
    action="store_true",
    default=False,
    help="Enable multi-scale training with varying image sizes. Applies random scaling to input images during training",
  )

  parser.add_argument(
    "--aug-cfg",
    nargs="*",
    default={},
    action=ParseKwargs,
    help="Additional augmentation configuration as key=value pairs. Example: --aug-cfg brightness=0.1 contrast=0.1",
  )

  # Model architecture
  parser.add_argument(
    "--model",
    type=str,
    default="RN50",
    help="Vision backbone architecture. Examples: RN50, RN101, ViT-B/32, ViT-L/14, EVA02-CLIP-B-16, EVA02-CLIP-L-14-336",
  )

  parser.add_argument(
    "--pretrained",
    default="",
    type=str,
    help="Pretrained model weights source. Use 'openai' for OpenAI CLIP weights, "
    "'eva' for EVA-CLIP weights, or path to custom checkpoint file",
  )

  parser.add_argument(
    "--pretrained-image",
    default=False,
    action="store_true",
    help="Load ImageNet pretrained weights for image tower backbone (if available and different from CLIP weights)",
  )

  parser.add_argument(
    "--embed-dim",
    type=int,
    default=768,
    help="Embedding dimension for image and text features. Must match the model architecture",
  )

  parser.add_argument(
    "--extract-type",
    type=str,
    choices=["v1", "v2"],
    default="v2",
    help="Feature extraction method version used in encode_pseudo_boxes. v2 is recommended for better performance",
  )

  parser.add_argument(
    "--force-quick-gelu",
    default=False,
    action="store_true",
    help="Force use of QuickGELU activation instead of standard GELU for non-OpenAI transformer models",
  )

  parser.add_argument(
    "--force-custom-text",
    default=False,
    action="store_true",
    help="Force use of CustomTextCLIP model with separate text tower instead of unified CLIP model",
  )

  parser.add_argument(
    "--force-patch-dropout",
    default=None,
    type=float,
    help="Override patch dropout rate during training (0.0-1.0). "
    "Useful for fine-tuning with reduced dropout near the end of training",
  )

  # Model freezing and fine-tuning
  parser.add_argument(
    "--lock-image",
    default=False,
    action="store_true",
    help="Freeze most of the image tower by disabling gradients. Used with lock-image-unlocked-groups to fine-tune only top layers",
  )

  parser.add_argument(
    "--lock-image-unlocked-groups",
    type=int,
    default=3,
    help="Number of image tower layer groups to keep unfrozen when --lock-image is used. "
    "For ViT-B/16: 12 layers total, for ViT-L/14: 24 layers total",
  )

  parser.add_argument(
    "--lock-image-freeze-bn-stats",
    default=True,
    action="store_true",
    help="Freeze BatchNorm running statistics in locked image tower layers",
  )

  # Training hyperparameters
  parser.add_argument("--epochs", type=int, default=32, help="Number of training epochs")

  parser.add_argument(
    "--batch-size", type=int, default=64, help="Batch size per GPU. Total effective batch size = batch_size × num_gpus"
  )

  parser.add_argument(
    "--lr", type=float, default=1e-5, help="Learning rate. Will be scaled by lr-scaling method in distributed training"
  )

  parser.add_argument(
    "--beta1",
    type=float,
    default=None,
    help="Adam optimizer beta1 parameter. If None, uses model-specific default (0.9 for both ViT and ResNet)",
  )

  parser.add_argument(
    "--beta2",
    type=float,
    default=None,
    help="Adam optimizer beta2 parameter. If None, uses model-specific default (0.98 for ViT, 0.999 for ResNet)",
  )

  parser.add_argument(
    "--eps",
    type=float,
    default=None,
    help="Adam optimizer epsilon parameter. If None, uses model-specific default (1e-6 for ViT, 1e-8 for ResNet)",
  )

  parser.add_argument("--wd", type=float, default=0.2, help="Weight decay (L2 regularization) strength")

  parser.add_argument("--warmup", type=int, default=10000, help="Number of warmup steps for learning rate schedule")

  parser.add_argument(
    "--lr-scheduler",
    type=str,
    default="cosine",
    help="Learning rate scheduler type: 'cosine' for cosine annealing, 'const' for constant, 'const-cooldown' for constant then cooldown",
  )

  parser.add_argument(
    "--lr-cooldown-end",
    type=float,
    default=0.0,
    help="Final learning rate for cooldown schedule (used with const-cooldown scheduler)",
  )

  parser.add_argument(
    "--lr-cooldown-power",
    type=float,
    default=1.0,
    help="Power for polynomial cooldown schedule (1.0 = linear decay, 2.0 = quadratic)",
  )

  parser.add_argument(
    "--skip-scheduler",
    action="store_true",
    default=False,
    help="Skip learning rate decay and use constant learning rate throughout training",
  )

  parser.add_argument(
    "--grad-clip-norm",
    type=float,
    default=None,
    help="Gradient clipping norm. Clips gradients if their norm exceeds this value (prevents gradient explosion)",
  )

  parser.add_argument(
    "--accum-freq",
    type=int,
    default=1,
    help="Gradient accumulation frequency. Updates model every N steps to simulate larger batch sizes",
  )

  # Loss function configuration
  parser.add_argument(
    "--alpha",
    type=float,
    default=2.0,
    help="Alpha parameter for loss weighting in CLIPSelf. Used in scripts: 0.7 for ViT-B/16, 0.95 for ViT-L/14",
  )

  parser.add_argument(
    "--kl-weight", type=float, default=1.0, help="Weight for KL divergence loss component in distillation training"
  )

  parser.add_argument(
    "--contrast-weight",
    type=float,
    default=1.0,
    help="Weight for contrastive loss component (used in RegionCLIP training)",
  )

  parser.add_argument(
    "--l1-weight", type=float, default=0.10, help="Weight for L1 regularization loss component in CLIPSelf training"
  )

  parser.add_argument(
    "--smooth-weight",
    type=float,
    default=0.0,
    help="Weight for smoothness regularization (encourages smooth feature transitions)",
  )

  parser.add_argument(
    "--cosine-weight", type=float, default=1.0, help="Weight for cosine similarity loss component in CLIPSelf training"
  )

  parser.add_argument(
    "--fix-logit-scale",
    action="store_true",
    default=False,
    help="Fix the logit scale parameter instead of learning it during training",
  )

  # Segmentation and embeddings
  parser.add_argument(
    "--val-segm-root",
    type=str,
    default="data/coco/annotations/panoptic_val2017",
    help="Directory containing validation panoptic segmentation masks",
  )

  parser.add_argument(
    "--train-segm-root",
    type=str,
    default="data/coco/annotations/panoptic_val2017",
    help="Directory containing training panoptic segmentation masks",
  )

  parser.add_argument(
    "--downsample-factor",
    type=int,
    default=16,
    help="Downsampling factor for segmentation masks to reduce memory usage during evaluation",
  )

  parser.add_argument(
    "--embed-path",
    type=str,
    default="metadata/coco_clip_hand_craft_RN50.npy",
    help="Path to precomputed text embeddings for evaluation class names (.npy file). "
    "Use metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy for EVA-CLIP ViT-B/16",
  )

  parser.add_argument(
    "--train-embed-path",
    type=str,
    default="",
    help="Path to precomputed text embeddings for training class names (required for region_clip mode)",
  )

  # Computational settings
  parser.add_argument(
    "--precision",
    choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
    default="amp",
    help="Floating point precision: 'amp' (automatic mixed precision), 'fp16', 'bf16', or 'fp32'",
  )

  parser.add_argument(
    "--grad-checkpointing",
    default=False,
    action="store_true",
    help="Enable gradient checkpointing to reduce memory usage at cost of computation time",
  )

  parser.add_argument(
    "--torchscript",
    default=False,
    action="store_true",
    help="Compile model with TorchScript for potential performance gains (experimental)",
  )

  parser.add_argument("--workers", type=int, default=1, help="Number of data loader worker processes per GPU")

  parser.add_argument(
    "--use-bn-sync",
    default=False,
    action="store_true",
    help="Use synchronized batch normalization across GPUs in distributed training",
  )

  parser.add_argument(
    "--gather-with-grad",
    default=False,
    action="store_true",
    help="Enable full distributed gradient for feature gathering (increases memory usage but may improve convergence)",
  )

  # Distributed training
  parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="URL for distributed training setup. 'env://' reads from environment variables (recommended)",
  )

  parser.add_argument(
    "--dist-backend",
    default="nccl",
    type=str,
    help="Distributed training backend: 'nccl' for GPUs (recommended), 'gloo' for CPUs",
  )

  parser.add_argument(
    "--horovod", default=False, action="store_true", help="Use Horovod instead of PyTorch native distributed training"
  )

  parser.add_argument(
    "--ddp-static-graph",
    default=False,
    action="store_true",
    help="Enable static graph optimization for DistributedDataParallel (requires PyTorch >= 1.11)",
  )

  parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't automatically set device from local rank (useful when CUDA_VISIBLE_DEVICES is manually set)",
  )

  # Checkpointing and evaluation
  parser.add_argument(
    "--cache-dir",
    type=str,
    default="checkpoints",
    help="Directory to cache/store model checkpoints and pretrained weights",
  )

  parser.add_argument("--save-frequency", type=int, default=1, help="Save checkpoint every N epochs")

  parser.add_argument(
    "--save-most-recent",
    action="store_true",
    default=False,
    help="Always save the most recent model as 'epoch_latest.pt' in addition to numbered checkpoints",
  )

  parser.add_argument(
    "--delete-previous-checkpoint",
    default=False,
    action="store_true",
    help="Delete previous checkpoint when saving a new one to save disk space",
  )

  parser.add_argument("--resume", default=None, type=str, help="Path to checkpoint file to resume training from")

  parser.add_argument(
    "--zeroshot-frequency", type=int, default=2, help="Run zero-shot evaluation on validation set every N epochs"
  )

  # Logging and debugging
  parser.add_argument(
    "--logs", type=str, default="./logs/", help="Directory to store TensorBoard logs and training outputs"
  )

  parser.add_argument(
    "--name", type=str, default=None, help="Experiment name for logging directory. If None, uses timestamp"
  )

  parser.add_argument(
    "--log-local",
    action="store_true",
    default=False,
    help="Log files on local master process only (not global master in multi-node setup)",
  )

  parser.add_argument("--log-every-n-steps", type=int, default=100, help="Log training metrics every N steps")

  parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Enable debug mode with additional logging and validation checks",
  )

  parser.add_argument(
    "--copy-codebase",
    default=False,
    action="store_true",
    help="Copy entire codebase to log directory for experiment reproducibility",
  )

  parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible results across runs")

  # Advanced experimental options
  parser.add_argument(
    "--image-ave-pool",
    action="store_true",
    default=False,
    help="Use average pooling for image features instead of default pooling (experimental feature)",
  )

  parser.add_argument(
    "--roi-teacher",
    action="store_true",
    default=False,
    help="Use ROI-based teacher model for distillation (experimental feature, not commonly used)",
  )

  parser.add_argument(
    "--del-dist-model",
    action="store_true",
    default=False,
    help="Delete distributed model after training to save memory (useful for large models)",
  )

  # Parse arguments and apply defaults
  parsed_args = parser.parse_args(args)

  # Apply model-specific default parameters if not explicitly set
  default_params = get_default_params(parsed_args.model)
  for name, val in default_params.items():
    if getattr(parsed_args, name) is None:
      setattr(parsed_args, name, val)

  return parsed_args
