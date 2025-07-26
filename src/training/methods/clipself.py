"""
CLIPSelf training implementation for self-supervised learning with CLIP models.

This module implements the CLIPSelf method which trains a student model to match
features extracted by a teacher model on image crops and regions.
"""

import argparse
import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPSelf:
  """
  CLIPSelf training class for self-supervised learning.

  This class implements the CLIPSelf training method where a student model learns
  to extract features from regions of interest (ROIs) that match features extracted
  by a teacher model from corresponding image crops.

  The training process involves:
  1. Processing input images with optional multiscale augmentation
  2. Filtering valid bounding boxes based on confidence scores
  3. Extracting features from image crops using the teacher model
  4. Extracting features from ROIs using the student model
  5. Computing cosine similarity loss between normalized features
  """

  def __call__(
    self,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    model: nn.Module,
    dist_model: nn.Module,
    loss: Any,  # Loss function (not used in current implementation)
    device: torch.device,
    cast_dtype: torch.dtype,
    distributed: bool,
    args: argparse.Namespace,
  ) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor]:
    """
    Execute one training step of CLIPSelf.

    Args:
        batch: Tuple containing (images, normed_boxes, image_crops)
            - images: Input images tensor of shape (B, C, H, W)
            - normed_boxes: Normalized bounding boxes of shape (B, N, 5)
              where last dim is [x1, y1, x2, y2, confidence]
            - image_crops: Pre-extracted image crops corresponding to boxes
        model: Student model for extracting ROI features
        dist_model: Teacher model (distributed) for extracting crop features
        loss: Loss function (unused in current implementation)
        device: Device to run computations on
        cast_dtype: Data type to cast tensors to
        distributed: Whether using distributed training
        args: Training arguments containing:
            - multiscale: Whether to apply multiscale augmentation
            - extract_type: Type of feature extraction for student model
            - cosine_weight: Weight for cosine similarity loss

    Returns:
        Tuple containing:
        - losses: Dictionary with 'loss_cosine' key
        - batch_size: Number of images in the batch
        - logit_scale: Exponential of the model's logit scale parameter
    """
    # Handle distributed training by unwrapping DataParallel/DistributedDataParallel modules
    if distributed:
      model = model.module
      dist_model = dist_model.module

    # Unpack batch data - note that texts are not paired with images in CLIPSelf
    images, normed_boxes, image_crops = batch

    # Move tensors to the specified device and data type
    images: torch.Tensor = images.to(device=device, dtype=cast_dtype, non_blocking=True)
    normed_boxes: torch.Tensor = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
    image_crops: torch.Tensor = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

    # Apply multiscale augmentation if enabled
    if args.multiscale:
      images = self._apply_multiscale_augmentation(images)

    # Filter valid bounding boxes and prepare ROI lists
    rois_list, valid_crops_list = self._filter_valid_boxes_and_crops(normed_boxes, image_crops)

    # Concatenate all valid crops for batch processing
    concatenated_image_crops: torch.Tensor = torch.cat(valid_crops_list)

    # Extract features using teacher model (no gradients needed)
    with torch.no_grad():
      teacher_crop_features = dist_model.encode_image(concatenated_image_crops, normalize=False)

    # Extract features using student model on ROIs
    student_roi_features = model.encode_pseudo_boxes(
      images, rois_list, normalize=False, extract_type=args.extract_type
    )

    # Normalize features for cosine similarity computation
    normalized_student_features: torch.Tensor = F.normalize(student_roi_features, dim=-1)
    normalized_teacher_features: torch.Tensor = F.normalize(teacher_crop_features, dim=-1)

    # Compute cosine similarity loss
    # Loss = 1 - cosine_similarity, where cosine_similarity = dot_product of normalized vectors
    cosine_similarity: torch.Tensor = (normalized_student_features * normalized_teacher_features).sum(-1)
    loss_cosine: torch.Tensor = 1.0 - cosine_similarity.mean()

    # Apply loss weighting from arguments
    weighted_loss_cosine: torch.Tensor = loss_cosine * args.cosine_weight
    losses: Dict[str, torch.Tensor] = {"loss_cosine": weighted_loss_cosine}

    return losses, len(images), model.logit_scale.exp()

  def _apply_multiscale_augmentation(self, images: torch.Tensor) -> torch.Tensor:
    """
    Apply multiscale augmentation to input images.

    Randomly resizes square images to different scales for data augmentation.
    Supports two predefined scale sets based on input image size.

    Args:
      images (torch.Tensor): Input images tensor of shape (B, C, H, W)

    Returns:
      Resized images tensor with the same batch and channel dimensions

    Raises:
      NotImplementedError: If input image size is not supported
    """
    current_height, current_width = images.shape[2:]
    assert current_height == current_width, "Images must be square for multiscale augmentation"

    # Define target sizes based on input image resolution
    if current_height == 1024:
      target_sizes: List[int] = [320, 640, 896, 1024]
    elif current_height == 896:
      target_sizes = [336, 448, 672, 896]
    else:
      raise NotImplementedError(
        f"Multiscale augmentation not implemented for image size {current_height}x{current_width}"
      )

    # Randomly select a target size and resize images
    target_size: int = random.choice(target_sizes)
    resized_images: torch.Tensor = F.interpolate(images, size=(target_size, target_size), mode="bilinear")

    return resized_images

  def _filter_valid_boxes_and_crops(
    self, normed_boxes: torch.Tensor, image_crops: torch.Tensor
  ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Filter bounding boxes and corresponding crops based on confidence threshold.

    Only keeps boxes with confidence > 0.5 and their corresponding image crops.

    Args:
      normed_boxes (torch.Tensor): Normalized bounding boxes of shape (B, N, 5)
                    where last dimension is [x1, y1, x2, y2, confidence]
      image_crops (torch.Tensor): Pre-extracted image crops corresponding to boxes

    Returns:
      Tuple containing:
      - rois_list: List of valid ROI coordinates (without confidence) per image
      - crops_list: List of valid image crops per image
    """
    rois_list: List[torch.Tensor] = []
    crops_list: List[torch.Tensor] = []

    # Process each image in the batch
    for bboxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
      # Filter boxes with confidence > 0.5 (last column contains confidence scores)
      valid_mask: torch.Tensor = bboxes_per_image[:, -1] > 0.5

      # Extract valid ROI coordinates (first 4 columns: x1, y1, x2, y2)
      valid_rois: torch.Tensor = bboxes_per_image[valid_mask, :4]
      valid_crops: torch.Tensor = crops_per_image[valid_mask]

      rois_list.append(valid_rois)
      crops_list.append(valid_crops)

    return rois_list, crops_list
