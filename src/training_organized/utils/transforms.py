"""
Custom transform utilities for CLIPSelf training.

This module provides custom image transforms used in CLIPSelf datasets.
"""

import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, InterpolationMode


class CustomRandomResize(nn.Module):
  """
  Custom random resize transform with configurable scale range.
  """

  def __init__(self, scale=(0.5, 2.0), interpolation=InterpolationMode.BILINEAR):
    """
    Initialize CustomRandomResize.

    Args:
        scale: Tuple of (min_scale, max_scale)
        interpolation: Interpolation mode
    """
    super().__init__()
    self.min_scale, self.max_scale = min(scale), max(scale)
    self.interpolation = interpolation

  def forward(self, img):
    """
    Apply random resize to image.

    Args:
        img: PIL Image or Tensor

    Returns:
        Resized image
    """
    if isinstance(img, torch.Tensor):
      height, width = img.shape[:2]
    else:
      width, height = img.size

    scale = random.uniform(self.min_scale, self.max_scale)
    new_size = [int(height * scale), int(width * scale)]
    img = F.resize(img, new_size, self.interpolation)

    return img


class CustomRandomCrop(RandomCrop):
  """
  Custom random crop that handles cases where target size is larger than image size.
  """

  def forward(self, img):
    """
    Apply random crop to image.

    Args:
        img: PIL Image or Tensor

    Returns:
        Cropped image
    """
    width, height = F.get_image_size(img)
    tar_h, tar_w = self.size

    # Ensure target size doesn't exceed image size
    tar_h = min(tar_h, height)
    tar_w = min(tar_w, width)

    i, j, h, w = self.get_params(img, (tar_h, tar_w))

    return F.crop(img, i, j, h, w)
