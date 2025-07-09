import argparse
import random
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class CLIPSelf:
  """
  Self-supervised learning class for CLIP models using self-distillation.

  This class implements a distillation loss between a student and teacher CLIP model,
  where the student learns to match teacher features for cropped regions.
  """

  def __call__(
    self,
    batch: Tuple[Tensor, Tensor, Tensor],
    student_model,
    teacher_model,
    device: torch.device,
    cast_dtype: torch.dtype,
    distributed: bool,
    args: argparse.Namespace,
  ) -> Tuple[Dict[str, Tensor], int, Tensor]:
    """
    Compute distillation loss between student and teacher models on region proposals.

    Args:
      batch (Tuple[Tensor, Tensor, Tensor]): Tuple containing (images, normalized_boxes, image_crops)
        - images: Input images [batch_size, channels, height, width]
        - normalized_boxes: Bounding boxes with confidence scores [batch_size, num_boxes, 5]
        - image_crops: Pre-cropped regions from images [total_crops, channels, height, width]
      student_model: Model being trained
      teacher_model: Pre-trained model providing supervision
      device (torch.device): Device to run computations on
      cast_dtype (torch.dtype): Data type for tensors
      distributed (bool): Whether using distributed training
      args (argparse.Namespace): Training arguments containing hyperparameters

    Returns:
      Tuple of (losses_dict, batch_size, logit_scale)
    """
    if distributed:
      student_model = student_model.module
      teacher_model = teacher_model.module

    input_images, normalized_boxes, region_crops = batch
    input_images: Tensor = input_images.to(device=device, dtype=cast_dtype, non_blocking=True)
    normalized_boxes: Tensor = normalized_boxes.to(device=device, dtype=cast_dtype, non_blocking=True)
    region_crops: Tensor = region_crops.to(device=device, dtype=cast_dtype, non_blocking=True)

    if args.multiscale:
      input_images = self._apply_multiscale_augmentation(input_images)

    # Filter valid regions and prepare data for feature extraction
    valid_roi_boxes, valid_region_crops = self._filter_valid_regions(
      normalized_boxes, region_crops, confidence_threshold=0.5
    )

    # Extract features
    with torch.no_grad():  # No gradients needed for teacher
      teacher_features: Tensor = teacher_model.encode_image(valid_region_crops, normalize=False)

    student_features: Tensor = student_model.encode_pseudo_boxes(
      input_images,
      valid_roi_boxes,
      normalize=False,
      extract_type=args.extract_type,
    )

    # Normalize features
    normalized_student_features: Tensor = F.normalize(student_features, dim=-1)
    normalized_teacher_features: Tensor = F.normalize(teacher_features, dim=-1)

    # Compute cosine similarity and convert to loss
    cosine_similarity: Tensor = (normalized_student_features * normalized_teacher_features).sum(-1)
    cosine_loss: Tensor = 1.0 - cosine_similarity.mean()

    losses: Dict[str, Tensor] = {"loss_cosine": cosine_loss * args.cosine_weight}

    return losses, len(input_images), student_model.logit_scale.exp()

  def _apply_multiscale_augmentation(self, images: Tensor) -> Tensor:
    """
    Apply multiscale augmentation by randomly resizing images.

    Args:
      images: Input images tensor

    Returns:
      Resized images tensor
    """
    current_height = images.shape[2]

    # Define target sizes based on current image height
    if current_height == 1024:
      target_sizes: List[int] = [320, 640, 896, 1024]
    elif current_height == 896:
      target_sizes = [336, 448, 672, 896]
    else:
      raise NotImplementedError(f"Multiscale not implemented for height {current_height}")

    # Randomly select a target size and resize
    selected_size: int = random.choice(target_sizes)
    return F.interpolate(images, size=selected_size, mode="bilinear")

  def _filter_valid_regions(
    self, normalized_boxes: Tensor, region_crops: Tensor, confidence_threshold: float = 0.5
  ) -> Tuple[List[Tensor], Tensor]:
    """
    Filter regions based on confidence scores and prepare for feature extraction.

    Args:
      normalized_boxes: Bounding boxes with confidence scores [batch_size, num_boxes, 5]
      region_crops: Pre-cropped image regions [total_crops, channels, height, width]
      confidence_threshold: Minimum confidence score to keep a region

    Returns:
      Tuple of (valid_roi_boxes_list, concatenated_valid_crops)
    """
    valid_roi_boxes_list = []
    valid_crops_list = []

    # Process each image's boxes and crops
    for boxes_per_image, crops_per_image in zip(normalized_boxes, region_crops):
      # Filter based on confidence score (last column)
      high_confidence_mask = boxes_per_image[:, -1] > confidence_threshold

      # Keep only high-confidence boxes (first 4 columns are coordinates)
      valid_boxes = boxes_per_image[high_confidence_mask, :4]
      valid_crops = crops_per_image[high_confidence_mask]

      valid_roi_boxes_list.append(valid_boxes)
      valid_crops_list.append(valid_crops)

    # Concatenate all valid crops across the batch
    concatenated_valid_crops = torch.cat(valid_crops_list)

    return valid_roi_boxes_list, concatenated_valid_crops


def demo():
  """
  Demo function for CLIPSelf class using GridDistillDataset.

  This demo shows how to:
  1. Load a dataset with grid-based crops
  2. Initialize student and teacher models (mocked for demo)
  3. Process a batch through CLIPSelf
  4. Display loss computation results
  """
  import argparse

  import torchvision.transforms as T

  from training_mert.data import GridDistillDataset

  # Parse command line arguments
  parser = argparse.ArgumentParser(description="CLIPSelf Demo")
  parser.add_argument("--annotation_file_path", required=True, help="Path to COCO annotation JSON")
  parser.add_argument("--image_root_path", required=True, help="Directory containing images")
  parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to process")
  parser.add_argument("--batch_size", type=int, default=2, help="Batch size for processing")
  parser.add_argument("--max_split", type=int, default=4, help="Maximum grid divisions per dimension")
  parser.add_argument("--max_boxes", type=int, default=16, help="Maximum number of boxes per image")
  parser.add_argument("--crop_size", type=int, default=224, help="Size of extracted crops")
  parser.add_argument("--cosine_weight", type=float, default=1.0, help="Weight for cosine loss")
  parser.add_argument("--extract_type", type=str, default="roi_align", help="Feature extraction type")
  parser.add_argument("--multiscale", action="store_true", help="Enable multiscale augmentation")

  args = parser.parse_args()

  print("CLIPSelf Demo")
  print("=" * 50)
  print(f"Batch size: {args.batch_size}")
  print(f"Cosine weight: {args.cosine_weight}")
  print(f"Multiscale: {args.multiscale}")
  print("-" * 50)

  # Setup device and data types
  cast_dtype = torch.float32

  # Setup transforms for full images and crops
  image_transform = T.Compose(
    [
      T.Resize(1024, interpolation=T.InterpolationMode.BICUBIC),
      T.CenterCrop(1024),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
  )

  crop_transform = T.Compose(
    [
      T.Resize(args.crop_size, interpolation=T.InterpolationMode.BICUBIC),
      T.CenterCrop(args.crop_size),
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
  )

  # Initialize dataset
  print("Initializing GridDistillDataset...")
  dataset = GridDistillDataset(
    annotations_file_path=args.annotation_file_path,
    image_root_path=args.image_root_path,
    transforms=[image_transform, crop_transform],
    max_split=args.max_split,
    crop_size=args.crop_size,
    max_boxes=args.max_boxes,
  )

  print(f"Dataset size: {len(dataset)}")
  print(f"Grid templates available: {len(dataset.box_templates)}")

  # Create a simple DataLoader for batching
  from torch.utils.data import DataLoader

  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

  # Mock student and teacher models for demo purposes
  class MockCLIPModel:
    """Mock CLIP model for demonstration purposes."""

    def __init__(self, feature_dim=512):
      self.feature_dim = feature_dim
      self.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)  # ln(100)

    def encode_image(self, images: Tensor, normalize: bool = True) -> Tensor:
      """Mock image encoding - returns random features."""
      batch_size = images.shape[0]
      features = torch.randn(batch_size, self.feature_dim, device=images.device, dtype=images.dtype)
      if normalize:
        features = F.normalize(features, dim=-1)
      return features

    def encode_pseudo_boxes(
      self, images: Tensor, roi_boxes_list, normalize: bool = True, extract_type: str = "roi_align"
    ) -> Tensor:
      """Mock ROI feature extraction - returns random features."""
      total_boxes = sum(len(boxes) for boxes in roi_boxes_list)
      features = torch.randn(total_boxes, self.feature_dim, device=images.device, dtype=images.dtype)
      if normalize:
        features = F.normalize(features, dim=-1)
      return features

  # Initialize models
  print("\nInitializing mock models...")
  student_model = MockCLIPModel(feature_dim=512)
  teacher_model = MockCLIPModel(feature_dim=512)

  # Initialize CLIPSelf
  clipself = CLIPSelf()

  print("\nProcessing batches...")
  print("-" * 30)

  # Process samples
  total_processed = 0
  for batch_idx, batch in enumerate(dataloader):
    if total_processed >= args.num_samples:
      break

    print(f"\nBatch {batch_idx + 1}:")

    # Extract batch components
    images, boxes, crops = batch
    print(f"  Images shape: {list(images.shape)}")
    print(f"  Boxes shape: {list(boxes.shape)}")
    print(f"  Crops shape: {list(crops.shape)}")

    # Count valid boxes per image
    valid_counts = []
    for i in range(boxes.shape[0]):
      valid_count = (boxes[i, :, 4] == 1.0).sum().item()
      valid_counts.append(valid_count)
    print(f"  Valid boxes per image: {valid_counts}")

    try:
      # Process through CLIPSelf
      losses, batch_size, logit_scale = clipself(
        batch=(images, boxes, crops),
        student_model=student_model,
        teacher_model=teacher_model,
        device="cpu",
        cast_dtype=cast_dtype,
        distributed=False,
        args=args,
      )

      # Display results
      print(f"  Batch size: {batch_size}")
      print(f"  Logit scale: {logit_scale.item():.4f}")
      print("  Losses:")
      for loss_name, loss_value in losses.items():
        print(f"    {loss_name}: {loss_value.item():.6f}")

      total_processed += batch_size

    except Exception as e:
      print(f"  Error processing batch: {e}")
      continue

  print(f"\nDemo completed! Processed {total_processed} samples.")
  print("\nNote: This demo uses mock models that generate random features.")
  print("In real usage, you would load pre-trained CLIP models for student and teacher.")


if __name__ == "__main__":
  demo()
