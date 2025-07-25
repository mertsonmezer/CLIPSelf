"""
Core CLIPSelf implementation.

This module contains the main CLIPSelf algorithm in a clean, well-documented format.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from training_organized.config import CLIPSelfConfig


class CLIPSelfMethod:
  """
  CLIPSelf method for self-supervised learning of region representations.

  This class implements the core CLIPSelf algorithm which trains a student model
  to match region features with crop features from a teacher model.
  """

  def __init__(self, config: CLIPSelfConfig):
    """
    Initialize CLIPSelf method.

    Args:
      config (CLIPSelfConfig): CLIPSelf configuration containing method parameters
    """
    self.config: CLIPSelfConfig = config
    self.logger: logging.Logger = logging.getLogger(__name__)

  def __call__(
    self,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    device: torch.device,
    cast_dtype: Optional[torch.dtype],
    distributed: bool,
  ) -> Tuple[Dict[str, torch.Tensor], int, torch.Tensor]:
    """
    Apply CLIPSelf training step.

    Args:
      batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of (images, normed_boxes, image_crops)
      student_model (torch.nn.Module): Student model to train
      teacher_model (Optional[torch.nn.Module]): Teacher model (frozen) for generating targets
      device (torch.device): Device to run computation on
      cast_dtype (Optional[torch.dtype]): Data type for mixed precision
      distributed (bool): Whether using distributed training

    Returns:
      Tuple of (losses_dict, batch_size, logit_scale)
    """
    # Unwrap models if using DDP
    if distributed:
      student_model = student_model.module
      teacher_model = teacher_model.module

    # Unpack batch
    images, normed_boxes, image_crops = batch

    # Move to device and cast dtype
    images: torch.Tensor = images.to(device=device, dtype=cast_dtype, non_blocking=True)
    normed_boxes: torch.Tensor = normed_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
    image_crops: torch.Tensor = image_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

    # Apply multiscale training if enabled
    images = self._apply_multiscale_if_enabled(images)

    # Prepare regions of interest
    rois_list, crops_list = self._prepare_rois_and_crops(normed_boxes, image_crops)

    # Get teacher features (no gradients)
    teacher_features: torch.Tensor = self._get_teacher_features(teacher_model, crops_list)

    # Get student features (with gradients)
    student_features: torch.Tensor = self._get_student_features(student_model, images, rois_list)

    # Compute CLIPSelf loss
    losses: Dict[str, torch.Tensor] = self._compute_clipself_loss(student_features, teacher_features)

    return losses, len(images), student_model.logit_scale.exp()

  def _apply_multiscale_if_enabled(self, images: torch.Tensor) -> torch.Tensor:
    """
    Apply multiscale training by randomly resizing images.

    Args:
      images (torch.Tensor): Input images tensor

    Returns:
      Potentially resized images
    """
    if not hasattr(self.config, "multiscale") or not self.config.multiscale:
      return images

    cur_h, cur_w = images.shape[2:]
    if cur_h != cur_w:
      self.logger.warning(f"Non-square images detected: {cur_h}x{cur_w}. Multiscale may not work correctly.")
      return images

    # Determine target sizes based on current resolution
    if cur_h == 1024:
      target_sizes = self.config.multiscale_targets_1024
    elif cur_h == 896:
      target_sizes = self.config.multiscale_targets_896
    else:
      self.logger.warning(f"Unsupported resolution for multiscale: {cur_h}. Skipping multiscale.")
      return images

    # Randomly select target size and resize
    target_size: int = random.choice(target_sizes)
    if target_size != cur_h:
      images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)
      self.logger.debug(f"Multiscale: resized from {cur_h}x{cur_w} to {target_size}x{target_size}")

    return images

  def _prepare_rois_and_crops(
    self, normed_boxes: torch.Tensor, image_crops: torch.Tensor
  ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Prepare regions of interest and corresponding crops from batch.

    Args:
      normed_boxes: Normalized bounding boxes [B, N, 5] (x1, y1, x2, y2, confidence)
      image_crops: Image crops corresponding to boxes [B, N, C, H, W]

    Returns:
      Tuple of (rois_list, crops_list) where each is a list of tensors per image
    """
    rois_list = []
    crops_list = []

    for boxes_per_image, crops_per_image in zip(normed_boxes, image_crops):
      # Filter boxes by confidence threshold
      valid_mask = boxes_per_image[:, -1] > 0.5

      valid_boxes = boxes_per_image[valid_mask, :4]  # Only box coordinates
      valid_crops = crops_per_image[valid_mask]

      rois_list.append(valid_boxes)
      crops_list.append(valid_crops)

    return rois_list, crops_list

  def _get_teacher_features(
    self, teacher_model: Optional[torch.nn.Module], crops_list: List[torch.Tensor]
  ) -> torch.Tensor:
    """
    Get teacher features from image crops.

    Args:
      teacher_model: Teacher model (frozen)
      crops_list: List of crops per image

    Returns:
      Teacher features tensor
    """
    if teacher_model is None:
      raise ValueError("Teacher model is required for CLIPSelf training")

    # Concatenate all crops
    all_crops: torch.Tensor = torch.cat(crops_list, dim=0)

    if len(all_crops) == 0:
      raise ValueError("No valid crops found in batch")

    # Get teacher features (no gradients)
    with torch.no_grad():
      teacher_features = teacher_model.encode_image(all_crops, normalize=False)

    return teacher_features

  def _get_student_features(
    self, student_model: torch.nn.Module, images: torch.Tensor, rois_list: List[torch.Tensor]
  ) -> torch.Tensor:
    """
    Get student features from regions of interest.

    Args:
      student_model: Student model to train
      images: Full images
      rois_list: List of ROIs per image

    Returns:
      Student features tensor
    """
    total_rois: int = sum(len(rois) for rois in rois_list)
    if total_rois == 0:
      raise ValueError("No valid ROIs found in batch")

    # Extract features using the student model
    student_features = student_model.encode_pseudo_boxes(
      images, rois_list, normalize=False, extract_type=self.config.extract_type
    )

    return student_features

  def _compute_clipself_loss(
    self, student_features: torch.Tensor, teacher_features: torch.Tensor
  ) -> Dict[str, torch.Tensor]:
    """
    Compute CLIPSelf cosine similarity loss.

    Args:
      student_features: Features from student model
      teacher_features: Features from teacher model

    Returns:
      Dictionary containing loss values
    """
    # Normalize features
    normed_student_features: torch.Tensor = F.normalize(student_features, dim=-1)
    normed_teacher_features: torch.Tensor = F.normalize(teacher_features, dim=-1)

    # Compute cosine similarity loss
    cosine_similarity: torch.Tensor = (normed_student_features * normed_teacher_features).sum(-1)
    loss_cosine: torch.Tensor = 1.0 - cosine_similarity.mean()

    # Apply weighting
    weighted_loss: torch.Tensor = loss_cosine * self.config.cosine_weight

    losses: Dict[str, Any] = {
      "loss_cosine": weighted_loss,
      "cosine_similarity": cosine_similarity.mean().detach(),  # For logging
    }

    return losses


def create_clipself_method(config: CLIPSelfConfig) -> CLIPSelfMethod:
  """
  Factory function to create CLIPSelf method.

  Args:
    config: CLIPSelf configuration

  Returns:
    CLIPSelf method instance
  """
  return CLIPSelfMethod(config)
