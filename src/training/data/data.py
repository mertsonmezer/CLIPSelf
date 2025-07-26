"""
Data loading and preprocessing module for CLIPSelf training.

This module contains dataset classes for different training modes:
- ProposalDistillDataset: For training with object proposals
- GridDistillDataset: For training with grid-based image patches
- COCOPanopticDataset: For training with panoptic segmentation data
- COCORegionCLIPDataset: For region-based CLIP training

All datasets support both local file system and Ceph distributed storage.
"""

import argparse
import io
import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from panopticapi import utils
from PIL import Image
from PIL.ImageFile import ImageFile
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, RandomHorizontalFlip

from open_clip.transform import ResizeLongest, get_scale
from training.data.coco_api import COCOPanoptic
from training.data.custom_transforms import CustomRandomCrop, CustomRandomResize
from training.utils.utils import mask2box

# Optional Ceph client for distributed storage
try:
  from petrel_client.client import Client
except ImportError:
  Client = None


class ProposalDistillDataset(Dataset[Tuple[torch.Tensor, ...]]):
  """
  Dataset for proposal-based distillation training.

  This dataset loads images and their corresponding object proposals (bounding boxes)
  from COCO-style annotations. It generates image crops from expanded bounding boxes
  for training region-level features.

  Args:
    input_filename (str): Path to COCO-style annotation file
    transforms (List[Any]): List of image transforms [image_transform, crop_transform]
    image_root (str): Root directory containing images
    crop_size (Union[int, Tuple[int, int]]): Size of cropped regions (default: 224)
    tokenizer (Any): Text tokenizer (optional, for compatibility)
    args (Any): Training arguments containing configuration parameters

  Returns:
      Tuple of (transformed_image, boxes_template, image_crops) where:
      - transformed_image: Preprocessed full image tensor
      - boxes_template: Tensor of shape (max_anns, 5) with [x1, y1, x2, y2, valid]
      - image_crops: Tensor of cropped image regions
  """

  def __init__(
    self,
    input_filename: str,
    transforms: List[Any],
    image_root: str,
    crop_size: Union[int, Tuple[int, int]] = 224,
    tokenizer: Any = None,
    args: argparse.Namespace = None,
  ):
    logging.debug(f"Loading COCO style data from {input_filename}.")
    self.coco = COCO(input_filename)
    logging.debug("Done loading data.")

    self.transforms: List[Any] = transforms
    self.tokenizer: Any = tokenizer
    self.image_root: str = image_root
    self.image_ids: List[int] = list(self.coco.imgs.keys())
    self.max_anns = 20  # Maximum number of annotations per image

    # Normalize crop_size to list format
    if not isinstance(crop_size, (tuple, list)):
      self.crop_size: List[int] = [crop_size, crop_size]
    else:
      self.crop_size: List[int] = list(crop_size)

    self.args: argparse.Namespace = args
    # Size constraints for valid proposals
    self.min_size: int = args.min_size
    self.max_size: int = args.max_size

    # Ceph distributed storage configuration
    self.ceph_root: str = args.train_ceph_root
    self.use_ceph: bool = self.ceph_root != ""
    self.file_client = None

  def read_image(self, image_name: str) -> Optional[Image.Image]:
    """
    Read image from either local filesystem or Ceph storage.

    Args:
      image_name (str): Name/path of the image file

    Returns:
      PIL Image object or None if loading fails
    """
    if self.use_ceph:
      image_path: str = os.path.join(self.ceph_root, image_name)
      if self.file_client is None:
        self.file_client = Client()
      try:
        img_bytes = self.file_client.get(image_path)
        buff = io.BytesIO(img_bytes)
        image = Image.open(buff)
      except Exception:
        print(f"Cannot load {image_path}", flush=True)
        return None
    else:
      image_path: str = os.path.join(self.image_root, image_name)
      try:
        image: ImageFile = Image.open(image_path)
      except Exception:
        print(f"Cannot load {image_path}", flush=True)
        return None

    width, height = image.size
    if width < 10 or height < 10:
      print(f"Invalid image, size {image.size}", flush=True)
      return None

    return image

  def __len__(self) -> int:
    return len(self.image_ids)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a training sample by index.

    Args:
      idx (int): Sample index

    Returns:
      Tuple of (image, boxes_template, image_crops)
    """
    image_id: int = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]

    # Extract image filename from annotation
    if "file_name" in image_info:
      image_name: str = image_info["file_name"]
    else:
      assert "coco_url" in image_info
      coco_url = image_info["coco_url"].split("/")
      image_name = os.path.join(coco_url[-2], coco_url[-1])

    # Load and validate image
    old_image = self.read_image(image_name)
    if old_image is None:
      # Fallback to random sample if image loading fails
      next_id = random.choice(range(self.__len__()))
      return self.__getitem__(next_id)

    img_w, img_h = old_image.width, old_image.height
    new_image = self.transforms[0](old_image)

    # Calculate scaling factor between original and transformed image
    scale = get_scale(old_image, new_image)
    anns = self.coco.imgToAnns[image_id]

    # Initialize output tensors
    boxes_template: torch.Tensor = torch.zeros(self.max_anns, 4 + 1)  # [x1, y1, x2, y2, valid]
    image_crops: torch.Tensor = torch.zeros(self.max_anns, 3, *self.crop_size)

    # Process annotations in random order
    indices: List[int] = list(range(len(anns)))
    random.shuffle(indices)
    num_valid_boxes = 0

    for i, ann_id in enumerate(indices[: self.max_anns]):
      ann = anns[ann_id]
      x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]

      # Filter by size constraints
      if w * h < (self.min_size**2) or w * h > (self.max_size**2):
        continue

      num_valid_boxes += 1

      # Expand bounding box by 50% for context
      cx, cy = x + w * 0.5, y + h * 0.5  # Box center
      x0, y0, x1, y1 = (
        max(cx - w * 0.75, 0),
        max(cy - h * 0.75, 0),
        min(cx + w * 0.75, img_w),
        min(cy + h * 0.75, img_h),
      )

      # Crop and transform the expanded region
      image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))

      # Store box coordinates in [x1, y1, x2, y2, valid] format
      box_info: torch.Tensor = torch.tensor([x, y, x + w, y + h, 1.0])
      boxes_template[i] = box_info

    # Handle edge case: no valid boxes found
    if num_valid_boxes == 0:
      boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])
      image_crops[0] = self.transforms[1](old_image.crop((0, 0, img_w // 4, img_h // 4)))

    # Normalize box coordinates to [0, 1] range
    _, h, w = new_image.shape
    boxes_template[:, :4] *= scale  # Apply scaling
    boxes_template[:, [0, 2]] /= w  # Normalize x coordinates
    boxes_template[:, [1, 3]] /= h  # Normalize y coordinates

    return new_image, boxes_template, image_crops


class GridDistillDataset(Dataset[Tuple[torch.Tensor, ...]]):
  """
  Dataset for grid-based distillation training.

  This dataset generates grid-based image patches by dividing images into regular
  grids and extracting crops from each grid cell. Used for training on spatial
  relationships and patch-level features.

  Args:
    input_filename: Path to COCO-style annotation file
    transforms: List of image transforms [image_transform, crop_transform]
    image_root: Root directory containing images
    max_split: Maximum grid size (default: 16)
    crop_size: Size of cropped regions (default: 224)
    pre_transforms: Whether to apply data augmentation (default: False)
    ceph_root: Ceph storage root path (default: "")
    args: Training arguments containing configuration parameters

  Returns:
    Tuple of (transformed_image, boxes_template, image_crops_template) where:
    - transformed_image: Preprocessed full image tensor
    - boxes_template: Tensor of shape (max_anns, 5) with [x1, y1, x2, y2, valid]
    - image_crops_template: Tensor of cropped grid regions
  """

  def __init__(
    self,
    input_filename: str,
    transforms: List[Any],
    image_root: str,
    max_split: int = 16,
    crop_size: Union[int, Tuple[int, int]] = 224,
    pre_transforms: bool = False,
    ceph_root: str = "",
    args: argparse.Namespace = None,
  ):
    self._init_choices(max_split)
    logging.debug(f"Loading COCO caption style data from {input_filename}.")
    self.coco = COCO(input_filename)
    logging.debug("Done loading data.")

    self.transforms: List[Any] = transforms
    self.image_root: str = image_root
    self.args: argparse.Namespace = args

    # Filter images based on training ratio
    image_ids: List[int] = list(self.coco.imgs.keys())
    train_ratio: float = args.train_ratio
    if train_ratio < 1.0:
      num_images = int(len(image_ids) * train_ratio)
      random.shuffle(image_ids)
      image_ids = image_ids[:num_images]
    self.image_ids: List[int] = image_ids

    self.max_anns: int = args.max_boxes

    # Normalize crop_size to list format
    if not isinstance(crop_size, (tuple, list)):
      self.crop_size: List[int] = [crop_size, crop_size]
    else:
      self.crop_size: List[int] = list(crop_size)

    self._init_boxes()

    # Ceph distributed storage configuration
    self.ceph_root = ceph_root
    self.use_ceph: bool = ceph_root != ""
    self.file_client = None

    # Optional pre-processing augmentations
    if pre_transforms:
      self.pre_transforms = Compose(
        [
          CustomRandomResize(scale=(0.5, 2.0)),
          CustomRandomCrop(size=self.transforms[0].transforms[0].max_size),
          RandomHorizontalFlip(),
        ]
      )
    else:
      self.pre_transforms = None

  def read_image(self, image_name: str) -> Optional[Image.Image]:
    """
    Read image from either local filesystem or Ceph storage.

    Args:
      image_name (str): Name/path of the image file

    Returns:
      PIL Image object or None if loading fails
    """
    if self.use_ceph:
      image_path: str = os.path.join(self.ceph_root, image_name)
      if self.file_client is None:
        self.file_client = Client()
      try:
        img_bytes = self.file_client.get(image_path)
        buff = io.BytesIO(img_bytes)
        image: ImageFile = Image.open(buff)
      except Exception:
        print(f"Cannot load {image_path}", flush=True)
        return None
    else:
      image_path: str = os.path.join(self.image_root, image_name)
      try:
        image: ImageFile = Image.open(image_path)
      except Exception:
        print(f"Cannot load {image_path}", flush=True)
        return None

    width, height = image.size
    if width < 10 or height < 10:
      print(f"Invalid image, size {image.size}", flush=True)
      return None

    return image

  def _init_choices(self, M: int = 16) -> None:
    """
    Initialize grid choices for different grid configurations.

    Args:
      M (int): Maximum grid size
    """
    choices: List[Tuple[int, int]] = []
    for m in range(1, M + 1):
      for n in range((m + 1) // 2, min(m * 2 + 1, M + 1)):
        choices.append((m, n))
    self.choices: List[Tuple[int, int]] = choices

  def __len__(self) -> int:
    return len(self.image_ids)

  def _init_boxes(self) -> None:
    """
    Initialize box templates for all grid configurations.
    Creates normalized coordinate grids for each (M, N) choice.
    """
    box_templates: Dict[Tuple[int, int], torch.Tensor] = {}
    for choice in self.choices:
      M, N = choice
      # Create grid coordinates
      grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1), indexing="xy")
      # Extract top-left and bottom-right corners
      x0y0s: torch.Tensor = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
      x1y1s: torch.Tensor = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
      # Combine into bounding boxes
      pseudo_boxes: torch.Tensor = torch.cat([x0y0s, x1y1s], dim=-1).view(-1, 4)

      assert pseudo_boxes.shape[0] == M * N
      box_templates[choice] = pseudo_boxes

    self.box_templates: Dict[Tuple[int, int], torch.Tensor] = box_templates

  def _obtain_image_crops(self, image: Image.Image, choice: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract image crops based on grid configuration.

    Args:
      image (Image.Image): PIL Image to crop
      choice (Tuple[int, int]): Grid configuration tuple (M, N)

    Returns:
      Tuple of (image_crops, boxes) tensors
    """
    image_crops: List[torch.Tensor] = []
    img_w, img_h = image.size
    normed_boxes: torch.Tensor = self.box_templates[choice]

    # Randomly select subset of boxes
    indices: List[int] = list(range(len(normed_boxes)))
    random.shuffle(indices)
    indices = indices[: self.max_anns]

    # Convert normalized coordinates to pixel coordinates
    boxes: torch.Tensor = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])

    for idx in indices:
      box: torch.Tensor = boxes[idx]
      x0, y0, x1, y1 = box.tolist()

      # Optionally expand crop region
      if self.args.crop_scale > 1.0:
        box_w, box_h = x1 - x0, y1 - y0
        cx, cy = (x1 + x0) / 2, (y1 + y0) / 2
        delta_factor = 0.5 * self.args.crop_scale
        x0, y0, x1, y1 = (
          max(cx - box_w * delta_factor, 0),
          max(cy - box_h * delta_factor, 0),
          min(cx + box_w * delta_factor, img_w),
          min(cy + box_h * delta_factor, img_h),
        )

      # Crop and transform the region
      image_crops.append(self.transforms[1](image.crop((x0, y0, x1, y1))))

    return torch.stack(image_crops), boxes[indices]

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a training sample by index.

    Args:
      idx (int): Sample index

    Returns:
      Tuple of (image, boxes_template, image_crops_template)
    """
    image_id: int = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]

    # Extract image filename
    if "file_name" in image_info:
      image_name: str = image_info["file_name"]
    else:
      assert "coco_url" in image_info
      coco_url = image_info["coco_url"].split("/")
      image_name = os.path.join(coco_url[-2], coco_url[-1])

    # Load and validate image
    old_image = self.read_image(image_name)
    if old_image is None:
      # Fallback to random sample if image loading fails
      next_id = random.choice(range(self.__len__()))
      return self.__getitem__(next_id)

    new_image = self.transforms[0](old_image)

    # Calculate scaling factor and get grid crops
    scale = get_scale(old_image, new_image)
    boxes_template: torch.Tensor = torch.zeros(self.max_anns, 4 + 1)  # [x1, y1, x2, y2, valid]
    image_crops_template: torch.Tensor = torch.zeros(self.max_anns, 3, *self.crop_size)

    # Get crops from randomly selected grid configuration
    image_crops, boxes = self._obtain_image_crops(old_image, random.choice(self.choices))
    assert image_crops.shape[0] == boxes.shape[0]

    # Normalize coordinates
    _, h, w = new_image.shape
    boxes[:, :4] *= scale
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h

    # Fill templates with actual data
    boxes_template[: boxes.shape[0], :4] = boxes
    boxes_template[: boxes.shape[0], 4] = 1.0
    image_crops_template[: boxes.shape[0]] = image_crops

    return new_image, boxes_template, image_crops_template


class COCOPanopticDataset(Dataset):
  """
  Dataset for COCO panoptic segmentation training.

  This dataset loads images with panoptic segmentation annotations, providing both
  bounding boxes and segmentation masks. Supports both 'thing' and 'stuff' categories.

  Args:
      input_filename: Path to COCO panoptic annotation file
      transforms: List of image transforms [image_transform, crop_transform]
      image_root: Root directory containing images
      embed_path: Path to precomputed text embeddings
      segm_root: Root directory containing segmentation masks
      crop_size: Size of cropped regions (default: 224)
      tokenizer: Text tokenizer (optional, for compatibility)
      downsample_factor: Factor for mask downsampling (default: 16)
      min_size: Minimum object size (default: 8)
      max_size: Maximum object size (default: 1024)

  Returns:
      Tuple of (image, boxes_template, image_crops, gt_masks, masked_image_crops) where:
      - image: Preprocessed full image tensor
      - boxes_template: Tensor of shape (max_anns, 8) with [x1, y1, x2, y2, class, valid, size, is_thing]
      - image_crops: Tensor of cropped image regions
      - gt_masks: Ground truth segmentation masks
      - masked_image_crops: Image crops with background masked out
  """

  def __init__(
    self,
    input_filename: str,
    transforms: Any,
    image_root: str,
    embed_path: str,
    segm_root: str,
    crop_size: Union[int, Tuple[int, int]] = 224,
    tokenizer: Any = None,
    downsample_factor: int = 16,
    min_size: int = 8,
    max_size: int = 1024,
  ):
    logging.debug(f"Loading COCO caption style data from {input_filename}.")
    self.coco = COCOPanoptic(input_filename)
    logging.debug("Done loading data.")

    self.transforms = transforms
    self.tokenize = tokenizer
    self.image_root = image_root
    self.embeddings = np.load(embed_path)
    self.image_ids = list(self.coco.imgs.keys())

    # Calculate maximum annotations per image
    num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
    self.max_anns = min(max(num_annos), 100)

    # Normalize crop_size to list format
    if not isinstance(crop_size, (tuple, list)):
      self.crop_size = [crop_size, crop_size]
    else:
      self.crop_size = list(crop_size)

    # Size constraints (fixed for validation)
    self.min_size = 8
    self.max_size = 1024

    # Segmentation configuration
    self.segm_root = segm_root
    self.downsample_factor = downsample_factor

    # Create transform for segmentation masks (downsampled)
    self.segm_transform = ResizeLongest(
      max_size=self.transforms[0].transforms[0].max_size // downsample_factor, fill=0
    )

    # Create category ID to label mapping
    cat_ids = sorted([cat["id"] for cat in self.coco.cats.values()])
    self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

  def __len__(self) -> int:
    return len(self.image_ids)

  @staticmethod
  def _load_segm(segm_path: str) -> np.ndarray:
    """
    Load panoptic segmentation map from file.

    Args:
        segm_path: Path to segmentation PNG file

    Returns:
        Segmentation map as numpy array with unique IDs per segment
    """
    segmentation = np.array(Image.open(segm_path), dtype=np.uint8)
    # Convert RGB segmentation to unique ID map
    segm_map = utils.rgb2id(segmentation)
    return segm_map

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a training sample by index.

    Args:
        idx: Sample index

    Returns:
        Tuple of (image, boxes_template, image_crops, gt_masks, masked_image_crops)
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
    new_image = self.transforms[0](old_image)

    # Calculate scaling factor
    scale = get_scale(old_image, new_image)
    anns = self.coco.imgToAnns[image_id]

    # Initialize output tensors
    # Box format: [x1, y1, x2, y2, class_label, valid, area, is_thing]
    boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1 + 1)
    image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
    gt_masks = torch.zeros(self.max_anns, self.segm_transform.max_size, self.segm_transform.max_size)
    masked_image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)

    for i, ann in enumerate(anns):
      if i == self.max_anns:
        break

      cat_id = ann["category_id"]
      is_thing = self.coco.cats[cat_id]["isthing"]

      # Extract bounding box based on category type
      if is_thing > 0:
        # For 'thing' categories, use provided bbox and expand
        x, y, w, h = ann["bbox"]
        cx, cy = x + w * 0.5, y + h * 0.5
        x0, y0, x1, y1 = (
          max(cx - w * 0.75, 0),
          max(cy - h * 0.75, 0),
          min(cx + w * 0.75, img_w),
          min(cy + h * 0.75, img_h),
        )
      else:
        # For 'stuff' categories, compute bbox from segmentation mask
        x0, y0, x1, y1 = mask2box(segm_map == ann["id"])
        x, y, w, h = x0, y0, x1 - x0, y1 - y0

      # Filter by size constraints
      if w * h < (self.min_size**2) or w * h > (self.max_size**2):
        continue

      # Generate regular image crop
      image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))

      # Generate masked image crop (background set to gray value 114)
      np_old_image = np.asarray(old_image.copy())
      np_old_image = np_old_image.copy()  # Make writable copy
      np_old_image[segm_map != ann["id"]] = 114  # Mask background
      masked_old_image = Image.fromarray(np_old_image)
      masked_image_crops[i] = self.transforms[1](masked_old_image.crop((x0, y0, x1, y1)))

      # Generate ground truth mask
      gt_mask = torch.from_numpy(segm_map == ann["id"]).float()
      gt_mask = self.segm_transform(gt_mask[None]) > 0.0

      # Prepare box information
      cls_label = self.cat_id2label[cat_id]
      box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h, is_thing])
      boxes_template[i] = box_info
      gt_masks[i] = gt_mask[0]

    # Normalize box coordinates to [0, 1] range
    _, h, w = new_image.shape
    boxes_template[:, :4] *= scale
    boxes_template[:, [0, 2]] /= w
    boxes_template[:, [1, 3]] /= h

    return new_image, boxes_template, image_crops, gt_masks, masked_image_crops


class COCORegionCLIPDataset(Dataset):
  """
  Dataset for COCO region-based CLIP training.

  This dataset loads images with object annotations for training region-level
  CLIP features. Focuses on learning visual-textual correspondences at the
  object/region level.

  Args:
      input_filename: Path to COCO-style annotation file
      transforms: List of image transforms [image_transform, crop_transform]
      image_root: Root directory containing images
      args: Training arguments containing configuration parameters

  Returns:
      Tuple of (transformed_image, boxes_template) where:
      - transformed_image: Preprocessed full image tensor
      - boxes_template: Tensor of shape (max_anns, 6) with [x1, y1, x2, y2, class, valid]
  """

  def __init__(self, input_filename: str, transforms: Any, image_root: str, args: Any):
    logging.debug(f"Loading COCO caption style data from {input_filename}.")
    self.coco = COCO(input_filename)
    logging.debug("Done loading data.")

    self.transforms = transforms
    self.image_root = image_root

    # Filter to images that have annotations
    image_ids = list(self.coco.imgToAnns.keys())
    train_ratio = args.train_ratio
    if train_ratio < 1.0:
      num_images = int(len(image_ids) * train_ratio)
      random.shuffle(image_ids)
      image_ids = image_ids[:num_images]
    self.image_ids = image_ids

    # Calculate maximum annotations per image
    num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
    self.max_anns = min(max(num_annos), 20)

    self.args = args

    # Ceph distributed storage configuration
    self.ceph_root = args.train_ceph_root
    self.use_ceph = self.ceph_root != ""
    self.file_client = None

    # Create category ID to label mapping
    cat_ids = sorted([cat["id"] for cat in self.coco.cats.values()])
    self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

  def __len__(self) -> int:
    return len(self.image_ids)

  def read_image(self, image_name: str) -> Image.Image:
    """
    Read image from either local filesystem or Ceph storage.

    Args:
        image_name: Name/path of the image file

    Returns:
        PIL Image object
    """
    if self.use_ceph:
      image_path = os.path.join(self.ceph_root, image_name)
      if self.file_client is None:
        self.file_client = Client()
      img_bytes = self.file_client.get(image_path)
      buff = io.BytesIO(img_bytes)
      image = Image.open(buff)
    else:
      image_path = os.path.join(self.image_root, image_name)
      image = Image.open(image_path)
    return image

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a training sample by index.

    Args:
        idx: Sample index

    Returns:
        Tuple of (image, boxes_template)
    """
    image_id = self.image_ids[idx]
    image_info = self.coco.imgs[image_id]
    image_name = image_info["file_name"]

    # Load and transform image
    old_image = self.read_image(image_name)
    new_image = self.transforms[0](old_image)

    # Calculate scaling factor and process annotations
    scale = get_scale(old_image, new_image)
    anns = self.coco.imgToAnns[image_id]

    # Initialize output tensor: [x1, y1, x2, y2, class_label, valid]
    boxes_template = torch.zeros(self.max_anns, 4 + 2)

    for i, ann in enumerate(anns):
      if i == self.max_anns:
        break

      cat_id = ann["category_id"]
      x, y, w, h = ann["bbox"]  # COCO format: [x, y, width, height]
      cls_label = self.cat_id2label[cat_id]

      # Store box info: [x1, y1, x2, y2, class_label, valid]
      box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0])
      boxes_template[i] = box_info

    # Normalize box coordinates to [0, 1] range
    _, h, w = new_image.shape
    boxes_template[:, :4] *= scale
    boxes_template[:, [0, 2]] /= w
    boxes_template[:, [1, 3]] /= h

    return new_image, boxes_template


def get_coco_panoptic_dataset(
  args: Any, preprocess_fn: Any, is_train: bool, epoch: int = 0, tokenizer: Any = None
) -> "DataInfo":
  """
  Create COCO panoptic dataset with DataLoader.

  Args:
      args: Training arguments
      preprocess_fn: Image preprocessing functions
      is_train: Whether this is training data
      epoch: Current epoch (for distributed sampling)
      tokenizer: Text tokenizer (optional)

  Returns:
      DataInfo object containing dataloader and sampler
  """
  input_filename = args.train_data if is_train else args.val_data
  assert input_filename

  dataset = COCOPanopticDataset(
    input_filename,
    preprocess_fn,
    segm_root=args.val_segm_root,
    image_root=args.val_image_root,
    embed_path=args.embed_path,
    tokenizer=tokenizer,
    crop_size=args.input_size,
    min_size=args.min_size,
    max_size=args.max_size,
    downsample_factor=args.downsample_factor,
  )
  num_samples = len(dataset)

  # Configure distributed sampling
  sampler = DistributedSampler(dataset) if args.distributed else None
  shuffle = is_train and sampler is None

  if is_train:
    batch_size = args.batch_size
  else:
    batch_size = min(args.batch_size, 1)  # Only support batch_size=1 for inference

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler,
    drop_last=is_train,
  )
  dataloader.num_samples = num_samples
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


def get_proposal_distill_dataset(
  args: Any, preprocess_fn: Any, is_train: bool, epoch: int = 0, tokenizer: Any = None
) -> "DataInfo":
  """
  Create proposal distillation dataset with DataLoader.

  Args:
      args: Training arguments
      preprocess_fn: Image preprocessing functions
      is_train: Whether this is training data
      epoch: Current epoch (for distributed sampling)
      tokenizer: Text tokenizer (optional)

  Returns:
      DataInfo object containing dataloader and sampler
  """
  assert is_train  # Only used for training
  input_filename = args.train_data
  assert input_filename

  dataset = ProposalDistillDataset(
    input_filename,
    preprocess_fn,
    image_root=args.train_image_root,
    tokenizer=tokenizer,
    crop_size=args.input_size,
    args=args,
  )
  num_samples = len(dataset)

  # Configure distributed sampling
  sampler = DistributedSampler(dataset) if args.distributed else None
  shuffle = is_train and sampler is None
  batch_size = args.batch_size

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler,
    drop_last=is_train,
  )
  dataloader.num_samples = num_samples
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


def get_grid_distill_dataset(
  args: Any, preprocess_fn: Any, is_train: bool, epoch: int = 0, tokenizer: Any = None
) -> "DataInfo":
  """
  Create grid distillation dataset with DataLoader.

  Args:
      args: Training arguments
      preprocess_fn: Image preprocessing functions
      is_train: Whether this is training data
      epoch: Current epoch (for distributed sampling)
      tokenizer: Text tokenizer (optional)

  Returns:
      DataInfo object containing dataloader and sampler
  """
  assert is_train  # Only used for training
  input_filename = args.train_data
  assert input_filename

  dataset = GridDistillDataset(
    input_filename=input_filename,
    transforms=preprocess_fn,
    image_root=args.train_image_root,
    crop_size=args.input_size,
    max_split=args.max_split,
    ceph_root=args.train_ceph_root,
    pre_transforms=args.pre_transforms,
    args=args,
  )
  num_samples = len(dataset)

  # Configure distributed sampling
  sampler = DistributedSampler(dataset) if args.distributed else None
  shuffle = is_train and sampler is None
  batch_size = args.batch_size

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler,
    drop_last=is_train,
  )
  dataloader.num_samples = num_samples
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


def get_region_clip_dataset(
  args: Any, preprocess_fn: Any, is_train: bool, epoch: int = 0, tokenizer: Any = None
) -> "DataInfo":
  """
  Create region CLIP dataset with DataLoader.

  Args:
      args: Training arguments
      preprocess_fn: Image preprocessing functions
      is_train: Whether this is training data
      epoch: Current epoch (for distributed sampling)
      tokenizer: Text tokenizer (optional)

  Returns:
      DataInfo object containing dataloader and sampler
  """
  assert is_train  # Only used for training
  input_filename = args.train_data
  assert input_filename

  dataset = COCORegionCLIPDataset(
    input_filename=input_filename,
    transforms=preprocess_fn,
    image_root=args.train_image_root,
    args=args,
  )
  num_samples = len(dataset)

  # Configure distributed sampling
  sampler = DistributedSampler(dataset) if args.distributed else None
  shuffle = is_train and sampler is None
  batch_size = args.batch_size

  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler,
    drop_last=is_train,
  )
  dataloader.num_samples = num_samples
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


class SharedEpoch:
  """
  Shared epoch counter for distributed training.

  Uses multiprocessing.Value to share epoch information across processes
  for consistent sampling behavior in distributed training.
  """

  def __init__(self, epoch: int = 0):
    self.shared_epoch = Value("i", epoch)

  def set_value(self, epoch: int) -> None:
    """Set the current epoch value."""
    self.shared_epoch.value = epoch

  def get_value(self) -> int:
    """Get the current epoch value."""
    return self.shared_epoch.value


@dataclass
class DataInfo:
  """
  Container for dataset information including dataloader and sampler.

  Args:
      dataloader: PyTorch DataLoader instance
      sampler: Distributed sampler (optional)
      shared_epoch: Shared epoch counter (optional)
  """

  dataloader: DataLoader
  sampler: Optional[DistributedSampler] = None
  shared_epoch: Optional[SharedEpoch] = None

  def set_epoch(self, epoch: int) -> None:
    """
    Set epoch for distributed sampling.

    Args:
        epoch: Current training epoch
    """
    if self.shared_epoch is not None:
      self.shared_epoch.set_value(epoch)
    if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
      self.sampler.set_epoch(epoch)


def get_dataset_fn(data_path: str, dataset_type: str):
  """
  Get dataset factory function based on dataset type.

  Args:
      data_path: Path to dataset (used for validation)
      dataset_type: Type of dataset to create

  Returns:
      Dataset factory function

  Raises:
      ValueError: If dataset type is not supported
  """
  if dataset_type == "coco_panoptic":
    return get_coco_panoptic_dataset
  elif dataset_type == "proposals_distill":
    return get_proposal_distill_dataset
  elif dataset_type == "grid_distill":
    return get_grid_distill_dataset
  elif dataset_type == "region_clip":
    return get_region_clip_dataset
  else:
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(
  args: Any, preprocess_fns: Tuple[Any, Any], epoch: int = 0, tokenizer: Any = None
) -> Dict[str, "DataInfo"]:
  """
  Create training and validation datasets based on configuration.

  Args:
      args: Training arguments containing dataset configuration
      preprocess_fns: Tuple of (train_preprocess, val_preprocess) functions
      epoch: Current training epoch
      tokenizer: Text tokenizer (optional)

  Returns:
      Dictionary containing 'train' and/or 'val' DataInfo objects
  """
  preprocess_train, preprocess_val = preprocess_fns
  data = {}

  if args.train_data:
    data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
      args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
    )

  if args.val_data:
    data["val"] = get_dataset_fn(args.val_data, dataset_type=args.test_type)(
      args, preprocess_val, is_train=False, tokenizer=tokenizer
    )

  return data


def demo():
  pass


if __name__ == "__main__":
  demo()
