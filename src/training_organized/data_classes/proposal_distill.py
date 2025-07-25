"""
Proposal Distillation Dataset for CLIPSelf training.

This module implements the ProposalDistillDataset which uses region proposals
for self-supervised learning.
"""

import logging
import random
import torch
from typing import List, Tuple
from pycocotools.coco import COCO
from open_clip.transform import get_scale

from training_organized.data_classes.base_dataset import BaseDataset


class ProposalDistillDataset(BaseDataset):
  """
  Dataset for proposal-based region distillation.

  Uses region proposals from COCO annotations to create image crops
  for CLIPSelf training.
  """

  def __init__(
    self, input_filename: str, transforms: List, image_root: str, crop_size: int = 224, tokenizer=None, args=None
  ):
    """
    Initialize ProposalDistillDataset.

    Args:
        input_filename: Path to COCO annotation file
        transforms: List of transforms [image_transform, crop_transform]
        image_root: Root directory for images
        crop_size: Size of cropped patches
        tokenizer: Text tokenizer (unused, kept for compatibility)
        args: Additional arguments
    """
    ceph_root = getattr(args, "train_ceph_root", "") if args else ""
    super().__init__(image_root, ceph_root, args)

    logging.debug(f"Loading COCO style data from {input_filename}.")
    self.coco = COCO(input_filename)
    logging.debug("Done loading data.")

    self.transforms = transforms
    self.tokenize = tokenizer
    self.image_ids = list(self.coco.imgs.keys())
    self.max_anns = 20

    # Handle crop size
    if not isinstance(crop_size, (tuple, list)):
      crop_size = [crop_size, crop_size]
    self.crop_size = crop_size

    # Get size constraints from args
    self.min_size = getattr(args, "min_size", 8) if args else 8
    self.max_size = getattr(args, "max_size", 1024) if args else 1024

  def __len__(self) -> int:
    """Return dataset length."""
    return len(self.image_ids)

  def _filter_valid_annotations(self, anns: List[dict], img_w: int, img_h: int) -> List[dict]:
    """
    Filter annotations based on size constraints.

    Args:
        anns: List of annotations
        img_w: Image width
        img_h: Image height

    Returns:
        List of valid annotations
    """
    valid_anns = []
    for ann in anns:
      x, y, w, h = ann["bbox"]
      area = w * h
      if self.min_size**2 <= area <= self.max_size**2:
        valid_anns.append(ann)
    return valid_anns

  def _create_expanded_crop_box(
    self, bbox: List[float], img_w: int, img_h: int, expansion_factor: float = 0.75
  ) -> Tuple[float, float, float, float]:
    """
    Create expanded bounding box for cropping.

    Args:
        bbox: Original bounding box [x, y, w, h]
        img_w: Image width
        img_h: Image height
        expansion_factor: Factor to expand the box

    Returns:
        Expanded box coordinates (x0, y0, x1, y1)
    """
    x, y, w, h = bbox
    cx, cy = x + w * 0.5, y + h * 0.5

    x0 = max(cx - w * expansion_factor, 0)
    y0 = max(cy - h * expansion_factor, 0)
    x1 = min(cx + w * expansion_factor, img_w)
    y1 = min(cy + h * expansion_factor, img_h)

    return x0, y0, x1, y1

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get dataset item.

    Args:
        idx: Item index

    Returns:
        Tuple of (full_image, boxes_template, image_crops)
    """
    image_id = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]
    image_name = self.get_image_name_from_info(image_info)

    # Load image
    old_image = self.read_image(image_name)
    if old_image is None:
      return self.handle_loading_failure(idx)

    img_w, img_h = old_image.width, old_image.height

    # Apply main image transform
    new_image = self.transforms[0](old_image)
    scale = get_scale(old_image, new_image)

    # Get annotations for this image
    anns = self.coco.imgToAnns[image_id]

    # Initialize templates
    boxes_template = torch.zeros(self.max_anns, 4 + 1)  # xyxy + score
    image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

    # Filter and shuffle annotations
    valid_anns = self._filter_valid_annotations(anns, img_w, img_h)
    random.shuffle(valid_anns)

    num_valid_boxes = 0
    for i, ann in enumerate(valid_anns[: self.max_anns]):
      x, y, w, h = ann["bbox"]

      # Create expanded crop box
      x0, y0, x1, y1 = self._create_expanded_crop_box([x, y, w, h], img_w, img_h)

      # Crop and transform image
      cropped_image = old_image.crop((x0, y0, x1, y1))
      image_crops[i] = self.transforms[1](cropped_image)

      # Store box information
      box_info = torch.tensor([x, y, x + w, y + h, 1.0])  # xyxy + score
      boxes_template[i] = box_info
      num_valid_boxes += 1

    # Handle case with no valid boxes
    if num_valid_boxes == 0:
      # Create a default box in the top-left corner
      default_w, default_h = img_w // 4, img_h // 4
      boxes_template[0] = torch.tensor([0, 0, default_w, default_h, 1.0])
      default_crop = old_image.crop((0, 0, default_w, default_h))
      image_crops[0] = self.transforms[1](default_crop)

    # Normalize boxes to image dimensions
    _, h, w = new_image.shape
    boxes_template[:, :4] *= scale
    boxes_template[:, [0, 2]] /= w  # Normalize x coordinates
    boxes_template[:, [1, 3]] /= h  # Normalize y coordinates

    return new_image, boxes_template, image_crops
