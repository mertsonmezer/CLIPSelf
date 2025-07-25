"""
Data configuration for CLIPSelf training.
"""

from dataclasses import dataclass
from typing import List, Optional

from training_organized.config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
  """Configuration for data loading and processing."""

  # Dataset settings
  dataset_type: str = "proposals_distill"  # proposals_distill, grid_distill
  train_data: Optional[str] = None
  val_data: Optional[str] = None
  test_type: str = "coco_panoptic"

  # Data paths
  imagenet_val: Optional[str] = None
  imagenet_v2: Optional[str] = None
  train_image_root: str = ""
  val_image_root: str = ""
  train_ceph_root: str = ""
  val_segm_root: str = ""
  train_segm_root: str = ""
  embed_path: str = ""
  train_embed_path: str = ""

  # Image processing
  crop_size: int = 224
  min_size: int = 800
  max_size: int = 1333
  multiscale: bool = False
  train_image_size: int = 1024

  # Data loading
  train_num_samples: Optional[int] = None
  val_num_samples: Optional[int] = None

  # Box/region settings
  max_boxes: int = 20
  max_masks: int = 20
  downsample_factor: int = 16

  # Data augmentation
  grid_noise: bool = False
  shift_range: float = 0.0
  scale_range: float = 0.0
  crop_scale: float = 1.0
  box_scale: float = 1.5
  pre_transforms: bool = False

  # Advanced settings
  embed_dim: int = 768
  fix_logit_scale: bool = False
  max_split: int = 6
  image_ave_pool: bool = False
  roi_teacher: bool = False
  mask_thr: float = 0.7
  train_ratio: float = 1.0
  del_dist_model: bool = False

  def validate(self) -> None:
    """Validate data configuration."""
    super().validate()

    valid_dataset_types: List[str] = ["proposals_distill", "grid_distill"]
    if self.dataset_type not in valid_dataset_types:
      raise ValueError(f"dataset_type must be one of {valid_dataset_types}")

    valid_test_types: List[str] = ["coco_panoptic"]
    if self.test_type not in valid_test_types:
      raise ValueError(f"test_type must be one of {valid_test_types}")

    if self.crop_size <= 0:
      raise ValueError("crop_size must be positive")

    if self.min_size <= 0:
      raise ValueError("min_size must be positive")

    if self.max_size <= 0:
      raise ValueError("max_size must be positive")

    if self.min_size > self.max_size:
      raise ValueError("min_size must be <= max_size")

    if self.max_boxes <= 0:
      raise ValueError("max_boxes must be positive")

    if self.max_masks <= 0:
      raise ValueError("max_masks must be positive")

    if self.downsample_factor <= 0:
      raise ValueError("downsample_factor must be positive")

    if self.train_image_size <= 0:
      raise ValueError("train_image_size must be positive")

    if self.embed_dim <= 0:
      raise ValueError("embed_dim must be positive")

    if self.max_split <= 0:
      raise ValueError("max_split must be positive")

    if not (0.0 <= self.mask_thr <= 1.0):
      raise ValueError("mask_thr must be between 0.0 and 1.0")

    if not (0.0 <= self.train_ratio <= 1.0):
      raise ValueError("train_ratio must be between 0.0 and 1.0")
