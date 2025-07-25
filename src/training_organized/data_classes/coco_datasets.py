"""
COCO Dataset implementations for CLIPSelf training.

This module implements COCO-specific datasets including panoptic segmentation.
"""

import logging
import os
import torch
import numpy as np
from typing import List, Tuple
from PIL import Image
from open_clip.transform import get_scale, ResizeLongest
from panopticapi import utils

from .base_dataset import BaseDataset
from ..utils.misc import mask2box
from ..coco_api import COCOPanoptic


class COCOPanopticDataset(BaseDataset):
  """
  Dataset for COCO panoptic segmentation with CLIPSelf training.

  Handles both thing and stuff categories with mask-based crops.
  """

  def __init__(
    self,
    input_filename: str,
    transforms: List,
    image_root: str,
    embed_path: str,
    segm_root: str,
    crop_size: int = 224,
    tokenizer=None,
    downsample_factor: int = 16,
    min_size: int = 8,
    max_size: int = 1024,
  ):
    """
    Initialize COCOPanopticDataset.

    Args:
        input_filename: Path to COCO panoptic annotation file
        transforms: List of transforms [image_transform, crop_transform]
        image_root: Root directory for images
        embed_path: Path to embeddings file
        segm_root: Root directory for segmentation masks
        crop_size: Size of cropped patches
        tokenizer: Text tokenizer
        downsample_factor: Downsampling factor for segmentation
        min_size: Minimum object size
        max_size: Maximum object size
    """
    super().__init__(image_root, "", None)  # No CEPH support for panoptic

    logging.debug(f"Loading COCO panoptic data from {input_filename}.")
    self.coco = COCOPanoptic(input_filename)
    logging.debug("Done loading data.")

    self.transforms = transforms
    self.tokenize = tokenizer
    self.embeddings = np.load(embed_path)
    self.image_ids = list(self.coco.imgs.keys())

    # Calculate max annotations
    num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
    self.max_anns = min(max(num_annos), 100)

    # Handle crop size
    if not isinstance(crop_size, (tuple, list)):
      crop_size = [crop_size, crop_size]
    self.crop_size = crop_size

    self.min_size = min_size
    self.max_size = max_size
    self.segm_root = segm_root
    self.downsample_factor = downsample_factor

    # Segmentation transform (downsampled)
    self.segm_transform = ResizeLongest(
      max_size=self.transforms[0].transforms[0].max_size // downsample_factor, fill=0
    )

    # Create category mapping
    cat_ids = sorted([cat["id"] for cat in self.coco.cats.values()])
    self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

  def __len__(self) -> int:
    """Return dataset length."""
    return len(self.image_ids)

  @staticmethod
  def _load_segm(segm_path: str) -> np.ndarray:
    """
    Load segmentation mask from file.

    Args:
        segm_path: Path to segmentation file

    Returns:
        Segmentation map as numpy array
    """
    segmentation = np.array(Image.open(segm_path), dtype=np.uint8)
    segm_map = utils.rgb2id(segmentation)
    return segm_map

  def _get_object_bbox(
    self, ann: dict, segm_map: np.ndarray, img_w: int, img_h: int
  ) -> Tuple[float, float, float, float]:
    """
    Get bounding box for object based on annotation type.

    Args:
        ann: Annotation dictionary
        segm_map: Segmentation map
        img_w: Image width
        img_h: Image height

    Returns:
        Bounding box coordinates (x0, y0, x1, y1)
    """
    cat_id = ann["category_id"]
    is_thing = self.coco.cats[cat_id]["isthing"]

    if is_thing > 0:
      # For thing categories, use bbox with expansion
      x, y, w, h = ann["bbox"]
      cx, cy = x + w * 0.5, y + h * 0.5
      x0 = max(cx - w * 0.75, 0)
      y0 = max(cy - h * 0.75, 0)
      x1 = min(cx + w * 0.75, img_w)
      y1 = min(cy + h * 0.75, img_h)
    else:
      # For stuff categories, use mask-based bbox
      x0, y0, x1, y1 = mask2box(segm_map == ann["id"])

    return x0, y0, x1, y1

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get dataset item.

    Args:
        idx: Item index

    Returns:
        Tuple of (full_image, boxes_template, image_crops, gt_masks, masked_image_crops)
    """
    image_id = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]
    image_name = image_info["file_name"]
    segm_file = image_info["segm_file"]

    # Load image and segmentation
    image_path = os.path.join(self.image_root, image_name)
    segm_path = os.path.join(self.segm_root, segm_file)

    segm_map = self._load_segm(segm_path)
    old_image = Image.open(image_path)
    img_w, img_h = old_image.width, old_image.height

    # Apply main image transform
    new_image = self.transforms[0](old_image)
    scale = get_scale(old_image, new_image)

    # Get annotations
    anns = self.coco.imgToAnns[image_id]

    # Initialize templates
    boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1 + 1)  # xyxy + cls + valid + size + isthing
    image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
    gt_masks = torch.zeros(self.max_anns, self.segm_transform.max_size, self.segm_transform.max_size)
    masked_image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

    for i, ann in enumerate(anns):
      if i == self.max_anns:
        break

      cat_id = ann["category_id"]
      is_thing = self.coco.cats[cat_id]["isthing"]

      # Get bounding box
      x0, y0, x1, y1 = self._get_object_bbox(ann, segm_map, img_w, img_h)
      x, y, w, h = x0, y0, x1 - x0, y1 - y0

      # Filter by size
      if w * h < (self.min_size**2) or w * h > (self.max_size**2):
        continue

      # Create regular image crop
      image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))

      # Create masked image crop
      np_old_image = np.asarray(old_image.copy())
      np_old_image[segm_map != ann["id"]] = 114  # Gray background
      masked_old_image = Image.fromarray(np_old_image)
      masked_image_crops[i] = self.transforms[1](masked_old_image.crop((x0, y0, x1, y1)))

      # Create ground truth mask
      gt_mask = torch.from_numpy(segm_map == ann["id"]).float()
      gt_mask = self.segm_transform(gt_mask[None]) > 0.0

      # Store annotation information
      cls_label = self.cat_id2label[cat_id]
      box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h, is_thing])
      boxes_template[i] = box_info
      gt_masks[i] = gt_mask[0]

    # Normalize boxes to image dimensions
    _, h, w = new_image.shape
    boxes_template[:, :4] *= scale
    boxes_template[:, [0, 2]] /= w  # Normalize x coordinates
    boxes_template[:, [1, 3]] /= h  # Normalize y coordinates

    return new_image, boxes_template, image_crops, gt_masks, masked_image_crops
