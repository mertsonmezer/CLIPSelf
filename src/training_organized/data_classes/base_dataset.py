"""
Base dataset class for CLIPSelf datasets.

This module provides the abstract base class and common utilities
for all CLIPSelf dataset implementations.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from torch.utils.data import Dataset

try:
  from petrel_client.client import Client
except ImportError:
  Client = None


class BaseDataset(Dataset[Tuple[torch.Tensor, ...]], ABC):
  """
  Abstract base class for all CLIPSelf datasets.

  Provides common functionality for image loading, CEPH support,
  and basic dataset operations.
  """

  def __init__(self, image_root: str, ceph_root: str = "", args: Optional[argparse.Namespace] = None):
    """
    Initialize base dataset.

    Args:
      image_root (str): Root directory for images
      ceph_root (str): CEPH root path (if using CEPH storage)
      args (argparse.Namespace, optional): Additional arguments
    """
    self.image_root: str = image_root
    self.ceph_root: str = ceph_root
    self.use_ceph: bool = ceph_root != ""
    self.FILE_CLIENT = None
    self.args: argparse.Namespace = args

  def read_image(self, image_name: str) -> Image.Image:
    """
    Read image from either local filesystem or CEPH storage.

    Args:
      image_name (str): Name/path of the image file

    Returns:
      PIL Image object, or None if loading failed
    """
    if self.use_ceph:
      image_path: str = os.path.join(self.ceph_root, image_name)
      if self.FILE_CLIENT is None:
        self.FILE_CLIENT = Client()
      try:
        img_bytes = self.FILE_CLIENT.get(image_path)
        buff = io.BytesIO(img_bytes)
        image: ImageFile = Image.open(buff)
      except Exception as e:
        logging.warning(f"Cannot load {image_path}: {e}")
        return None
    else:
      image_path = os.path.join(self.image_root, image_name)
      try:
        image = Image.open(image_path)
      except Exception as e:
        logging.warning(f"Cannot load {image_path}: {e}")
        return None

    # Validate image size
    width, height = image.size
    if width < 10 or height < 10:
      logging.warning(f"Invalid image size {image.size}")
      return None

    return image

  def get_image_name_from_info(self, image_info: Dict[str, Any]) -> str:
    """
    Extract image filename from COCO image info.

    Args:
      image_info (Dict[str, Any]): COCO image information dictionary

    Returns:
        Image filename/path
    """
    if "file_name" in image_info:
      return image_info["file_name"]
    elif "coco_url" in image_info:
      coco_url: str = image_info["coco_url"].split("/")
      return os.path.join(coco_url[-2], coco_url[-1])
    else:
      raise ValueError("No valid image path found in image_info")

  def handle_loading_failure(self, idx: int):
    """
    Handle image loading failure by returning a random valid sample.

    Args:
      idx (int): Current index that failed

    Returns:
      Result from a random valid sample
    """
    next_id: int = random.choice(range(len(self)))
    return self.__getitem__(next_id)

  @abstractmethod
  def __len__(self) -> int:
    """Return dataset length."""
    pass

  @abstractmethod
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
    """Get dataset item by index."""
    pass
