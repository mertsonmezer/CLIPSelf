"""
Grid Distillation Dataset for CLIPSelf training.

This module implements the GridDistillDataset which creates grid-based
image patches for self-supervised learning.
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import Compose, RandomHorizontalFlip

from open_clip.transform import get_scale

from ..utils.transforms import CustomRandomCrop, CustomRandomResize
from .base_dataset import BaseDataset


class GridDistillDataset(BaseDataset):
  """
  Dataset for grid-based image patch distillation.

  Creates grid-based patches from images for CLIPSelf training.
  Supports various grid configurations and augmentations.
  """

  def __init__(
    self,
    input_filename: str,
    transforms: List[Any],
    image_root: str,
    max_split: int = 16,
    crop_size: int | List[int] | Tuple[int, int] = 224,
    pre_transforms: bool = False,
    ceph_root: str = "",
    args: Optional[argparse.Namespace] = None,
  ):
    """
    Initialize GridDistillDataset.

    Args:
      input_filename (str): Path to COCO annotation file
      transforms (List[Any]): List of transforms [image_transform, crop_transform]
      image_root (str): Root directory for images
      max_split (int): Maximum grid split size
      crop_size (int | List[int] | Tuple[int, int]): Size of cropped patches
      pre_transforms (bool): Whether to use pre-transforms
      ceph_root (str): CEPH root path
      args (Optional[argparse.Namespace]): Additional arguments
    """
    super().__init__(image_root, ceph_root, args)

    self._init_grid_choices(max_split)

    logging.debug(f"Loading COCO caption style data from {input_filename}.")
    self.coco = COCO(input_filename)
    logging.debug("Done loading data.")

    self.transforms: List[Any] = transforms

    # Filter image IDs based on train ratio if specified
    image_ids: List[int] = list(self.coco.imgs.keys())
    train_ratio: float = getattr(args, "train_ratio", 1.0)
    if train_ratio < 1.0:
      num_images = int(len(image_ids) * train_ratio)
      random.shuffle(image_ids)
      image_ids = image_ids[:num_images]
    self.image_ids: List[int] = image_ids

    self.max_anns: int = getattr(args, "max_boxes", 20)

    # Handle crop size
    if not isinstance(crop_size, (tuple, list)):
      crop_size = [crop_size, crop_size]
    self.crop_size: List[int] | Tuple[int, int] = crop_size

    self._init_box_templates()

    # Setup pre-transforms if enabled
    if pre_transforms:
      self.pre_transforms: Compose | None = Compose(
        [
          CustomRandomResize(scale=(0.5, 2.0)),
          CustomRandomCrop(size=self.transforms[0].transforms[0].max_size),
          RandomHorizontalFlip(),
        ]
      )
    else:
      self.pre_transforms = None

  def _init_grid_choices(self, M: int = 16) -> None:
    """
    Initialize possible grid configurations.

    Args:
      M (int): Maximum grid size
    """
    choices: List[Tuple[int, int]] = []
    for m in range(1, M + 1):
      for n in range((m + 1) // 2, min(m * 2 + 1, M + 1)):
        choices.append((m, n))
    self.choices: List[Tuple[int, int]] = choices

  def _init_box_templates(self) -> None:
    """Initialize box templates for each grid choice."""
    box_templates: Dict[Tuple[int, int], torch.Tensor] = {}
    for choice in self.choices:
      M, N = choice
      grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1), indexing="xy")
      x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
      x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
      pseudo_boxes: torch.Tensor = torch.cat([x0y0s, x1y1s], dim=-1).view(-1, 4)

      assert pseudo_boxes.shape[0] == M * N
      box_templates[choice] = pseudo_boxes

    self.box_templates: Dict[Tuple[int, int], torch.Tensor] = box_templates

  def _obtain_image_crops(self, image: Image.Image, choice: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Obtain image crops based on grid choice.

    Args:
      image (Image.Image): PIL Image
      choice (Tuple[int, int]): Grid configuration (M, N)

    Returns:
      Tuple of (image_crops, boxes)
    """
    image_crops: List[torch.Tensor] = []
    img_w, img_h = image.size
    normed_boxes: torch.Tensor = self.box_templates[choice]

    # Randomly select boxes up to max_anns
    indices: List[int] = list(range(len(normed_boxes)))
    random.shuffle(indices)
    indices = indices[: self.max_anns]

    # Convert normalized boxes to pixel coordinates
    boxes: torch.Tensor = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])

    for idx in indices:
      box: torch.Tensor = boxes[idx]
      x0, y0, x1, y1 = box.tolist()

      # Apply crop scaling if specified
      if hasattr(self.args, "crop_scale") and self.args.crop_scale > 1.0:
        box_w, box_h = x1 - x0, y1 - y0
        cx, cy = (x1 + x0) / 2, (y1 + y0) / 2
        delta_factor: float = 0.5 * self.args.crop_scale
        x0 = max(cx - box_w * delta_factor, 0)
        y0 = max(cy - box_h * delta_factor, 0)
        x1 = min(cx + box_w * delta_factor, img_w)
        y1 = min(cy + box_h * delta_factor, img_h)

      cropped_image: Image.Image = image.crop((x0, y0, x1, y1))
      image_crops.append(self.transforms[1](cropped_image))

    return torch.stack(image_crops), boxes[indices]

  def __len__(self) -> int:
    """Return dataset length."""
    return len(self.image_ids)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get dataset item.

    Args:
      idx (int): Item index

    Returns:
      Tuple of (full_image, boxes_template, image_crops_template)
    """
    image_id: int = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]
    image_name: str = self.get_image_name_from_info(image_info)

    # Load image
    old_image: Image.Image = self.read_image(image_name)
    if old_image is None:
      return self.handle_loading_failure(idx)

    # Apply pre-transforms if enabled
    if self.pre_transforms is not None:
      old_image = self.pre_transforms(old_image)

    # Apply main image transform
    new_image: torch.Tensor = self.transforms[0](old_image)

    # Get scale factor
    scale: float = get_scale(old_image, new_image)

    # Initialize templates
    boxes_template: torch.Tensor = torch.zeros(self.max_anns, 4 + 1)  # xyxy + score
    image_crops_template: torch.Tensor = torch.zeros(self.max_anns, 3, *self.crop_size)

    # Get image crops and boxes
    image_crops, boxes = self._obtain_image_crops(old_image, random.choice(self.choices))

    assert image_crops.shape[0] == boxes.shape[0]

    # Normalize boxes to image dimensions
    _, h, w = new_image.shape
    boxes[:, :4] *= scale
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h

    # Fill templates
    num_boxes = boxes.shape[0]
    boxes_template[:num_boxes, :4] = boxes
    boxes_template[:num_boxes, 4] = 1.0  # Set score to 1.0
    image_crops_template[:num_boxes] = image_crops

    return new_image, boxes_template, image_crops_template
