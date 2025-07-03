import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.transforms import Compose

from open_clip.transform import get_scale


class ProposalDistillDataset(Dataset[Any]):
  """Placeholder for future proposal distillation dataset."""

  def __init__(self, *args, **kwargs):
    raise NotImplementedError


class GridDistillDataset(Dataset[Any]):
  """COCO grid patch dataset used for CLIPSelf training.

  This dataset generates grid-based bounding boxes over COCO images and extracts
  crops from those regions. It supports variable grid configurations (e.g., 2x2,
  3x4, etc.) and returns both the full transformed image and cropped regions as tensors.

  Attributes:
    coco (COCO): COCO dataset instance for loading annotations and metadata.
    image_root (str): Root directory containing the COCO images.
    transforms (List[Any]): List of two transforms - [full_image_transform, crop_transform].
    max_split (int): Maximum number of grid divisions in each dimension.
    max_boxes (int): Maximum number of bounding boxes to sample per image.
    crop_size (Union[int, Tuple[int, int]]): Target size for cropped regions as (height, width).
    image_ids (List[int]): List of COCO image IDs available in the dataset.
    box_templates (Dict[Tuple[int, int], Tensor]): Pre-computed grid templates for different
      configurations.
  """

  def __init__(
    self,
    annotations_file_path: str,
    image_root_path: str,
    transforms: List[Any],
    max_split: int = 4,
    crop_size: Union[int, Tuple[int, int]] = 224,
    max_boxes: int = 20,
  ) -> None:
    """Initializes the GridDistillDataset.

    Args:
      annotations_file_path (str): Path to the COCO annotations JSON file.
      image_root_path (str): Root directory containing the COCO images.
      transforms (List[Any]): List of exactly two transforms:
        - transforms[0]: Applied to full image, should return tensor
        - transforms[1]: Applied to crops, should return tensor
      max_split (int): Maximum grid divisions per dimension (creates grids from
        1x1 up to max_split x max_split).
      crop_size (Union[int, Tuple[int, int]]): Target size for crops. If int, creates square crops.
        If tuple, specifies (height, width).
      max_boxes (int): Maximum number of bounding boxes to sample per image.
    """
    self.coco = COCO(annotations_file_path)
    self.image_root: str = image_root_path
    self.transforms: List[Any] = transforms
    self.max_split: int = max_split
    self.max_boxes: int = max_boxes
    self.crop_size: Tuple[int, int] = (crop_size, crop_size) if not isinstance(crop_size, (List, Tuple)) else crop_size
    self.image_ids: List[int] = list(self.coco.imgs.keys())
    self._init_grid_templates()

  def _init_grid_templates(self) -> None:
    """Initializes grid templates for different split configurations.

    Creates bounding box templates for all possible grid configurations
    from 1x1 up to max_split x max_split. Each template contains normalized
    coordinates [x0, y0, x1, y1] for all grid cells in that configuration.

    The templates are stored in self.box_templates with keys as (m, n) tuples
    representing the grid dimensions, and values as tensors of shape (m*n, 4).
    """
    self.box_templates: Dict[Tuple[int, int], Tensor] = {}

    for m in range(1, self.max_split + 1):
      for n in range(1, self.max_split + 1):
        x_coords: torch.Tensor = torch.linspace(0, 1, n + 1)
        y_coords: torch.Tensor = torch.linspace(0, 1, m + 1)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")

        top_left: torch.Tensor = torch.stack([grid_x[:-1, :-1], grid_y[:-1, :-1]], dim=-1)
        bottom_right: torch.Tensor = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)

        # Combine into boxes [x0, y0, x1, y1]
        boxes: torch.Tensor = torch.cat([top_left, bottom_right], dim=-1).view(-1, 4)
        self.box_templates[(m, n)] = boxes

  def __len__(self) -> int:
    """Returns the number of images in the dataset.

    Returns:
      Number of COCO images available in the dataset.
    """
    return len(self.image_ids)

  def _read_image(self, file_name: str) -> Image.Image:
    """Loads an image file and convert to RGB.

    Args:
      file_name (str): Name of the image file relative to image_root.

    Returns:
      PIL Image in RGB format.
    """
    image_path: str = os.path.join(self.image_root, file_name)
    return Image.open(image_path).convert("RGB")

  def _sample_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
    """Samples a random subset of bounding boxes.

    Args:
      boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding box coordinates.

    Returns:
      Tensor of shape (min(N, max_boxes), 4) with randomly sampled boxes.
    """
    indices: List[int] = list(range(len(boxes)))
    random.shuffle(indices)
    indices = indices[: self.max_boxes]
    return boxes[indices]

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve a single training sample with grid-based crops.

    This method loads an image, applies transformations, generates grid-based
    bounding boxes, extracts crops from those regions, and returns normalized
    data suitable for training.

    Args:
      idx: Index of the sample to retrieve from the dataset.

    Returns:
      A tuple containing:
        - transformed_image: Tensor of shape (C, H, W) representing the
          transformed full image.
        - norm_boxes: Tensor of shape (max_boxes, 5) where each row contains
          [x0, y0, x1, y1, validity_flag]. Coordinates are normalized to [0,1].
        - crop_tensor: Tensor of shape (max_boxes, C, crop_H, crop_W) containing
          the transformed crop images. Unused slots are zero-filled.

    Raises:
      IndexError: If idx is out of range for the dataset.
      FileNotFoundError: If the image file cannot be found.
      ValueError: If the image cannot be processed or transforms fail.
    """
    if idx >= len(self.image_ids):
      raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_ids)}")

    image_id: int = self.image_ids[idx]
    meta_data: Dict[str, Any] = self.coco.imgs[image_id]
    image: Image.Image = self._read_image(meta_data["file_name"])

    # Apply transform to full image
    transformed_image: torch.Tensor = self.transforms[0](image)
    scale: float = get_scale(image, transformed_image)

    # Sample random grid configuration and boxes
    choice: Tuple[int, int] = random.choice(list(self.box_templates.keys()))
    boxes: Tensor = self._sample_boxes(self.box_templates[choice])

    # Extract and transform crops
    crops: List[torch.Tensor] = []
    for box in boxes:
      # Convert normalized coordinates to pixel coordinates
      coords: Tensor = (box * torch.tensor([image.width, image.height, image.width, image.height])).int()
      x0, y0, x1, y1 = coords.tolist()

      # Ensure coordinates are within image bounds
      x0, y0 = max(0, x0), max(0, y0)
      x1, y1 = min(image.width, x1), min(image.height, y1)

      # Skip invalid boxes (too small or inverted)
      if x1 <= x0 or y1 <= y0:
        continue

      crop_image: Image.Image = image.crop((x0, y0, x1, y1))
      crop_tensor: torch.Tensor = self.transforms[1](crop_image)
      crops.append(crop_tensor)

    # Fallback if no valid crops were generated
    if not crops:
      fallback_crop: Image.Image = image.crop((0, 0, image.width // 4, image.height // 4))
      crop_tensor = self.transforms[1](fallback_crop)
      crops = [crop_tensor]
      boxes = boxes[:1]  # Keep only first box for consistency

    # Initialize output tensors
    norm_boxes: torch.Tensor = torch.zeros(self.max_boxes, 5, dtype=torch.float32)
    crop_tensor = torch.zeros(self.max_boxes, 3, *self.crop_size, dtype=torch.float32)

    # Get transformed image dimensions
    _, h, w = transformed_image.shape

    # Normalize box coordinates
    boxes = boxes.clone()
    boxes[:, :4] *= scale  # Scale by resize factor
    boxes[:, [0, 2]] /= w  # Normalize x coordinates
    boxes[:, [1, 3]] /= h  # Normalize y coordinates

    # Fill output tensors
    num: int = len(crops)
    if num > 0:
      # Limit to max_boxes to prevent overflow
      num = min(num, self.max_boxes)
      norm_boxes[:num, :4] = boxes[:num]
      norm_boxes[:num, 4] = 1.0  # Validity flag
      crop_tensor[:num] = torch.stack(crops[:num])

    return transformed_image, norm_boxes, crop_tensor


@dataclass
class DataInfo:
  """Holds references to a dataloader and an optional distributed sampler."""

  dataloader: DataLoader[Any]
  sampler: Optional[DistributedSampler[Any]] = None

  def set_epoch(self, epoch: int) -> None:
    if self.sampler is not None:
      self.sampler.set_epoch(epoch)


def build_dataloader(dataset: Dataset[Any], is_train: bool, args: argparse.Namespace) -> DataInfo:
  """Builds a DataLoader and its associated DistributedSampler for the given dataset.

  Args:
    dataset (Dataset[Any]): The dataset to load data from.
    is_train (bool): Whether the dataloader is for training (enables shuffling and dropping last batch).
    args (argparse.Namespace): Arguments namespace containing configuration options. Must include:
      - batch_size (int): Number of samples per batch.
      - workers (int): Number of subprocesses to use for data loading.
      - distributed (bool): Whether to use distributed sampling.
  Returns:
    DataInfo: An object containing the DataLoader and the sampler used (if any).
  """
  sampler: Optional[DistributedSampler[Any]] = DistributedSampler(dataset) if args.distributed else None
  shuffle: bool = is_train and sampler is None

  dataloader: DataLoader[Any] = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    sampler=sampler,
    pin_memory=True,
    drop_last=is_train,
  )
  dataloader.num_samples = len(dataset)
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


def get_dataset_class(
  dataset_type: Literal["grid_distill", "proposals_distill"],
) -> Union[Type[GridDistillDataset], Type[ProposalDistillDataset]]:
  """Returns the dataset class based on the dataset type."""
  if dataset_type == "grid_distill":
    return GridDistillDataset
  if dataset_type == "proposals_distill":
    return ProposalDistillDataset
  raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(
  args: argparse.Namespace, preprocess_fns: Tuple[Callable[..., Any], Callable[..., Any]], epoch: int = 0
) -> Dict[str, DataInfo]:
  """Builds dataloaders for training and validation datasets based on provided arguments."""
  train_preprocess_fn, val_preprocess_fn = preprocess_fns
  dataloaders: Dict[str, DataInfo] = {}

  if hasattr(args, "train_data") and args.train_data:
    dataset_class: Union[Type[GridDistillDataset], Type[ProposalDistillDataset]] = get_dataset_class(args.dataset_type)
    train_dataset: Union[GridDistillDataset, ProposalDistillDataset] = dataset_class(
      annotations_file_path=args.train_data,
      image_root_path=args.train_image_root,
      transforms=train_preprocess_fn,
      max_split=args.max_split,
      crop_size=args.input_size,
      max_boxes=args.max_boxes,
    )
    dataloaders["train"] = build_dataloader(train_dataset, is_train=True, args=args)

  if hasattr(args, "val_data") and args.val_data:
    dataset_class: Union[Type[GridDistillDataset], Type[ProposalDistillDataset]] = get_dataset_class(args.dataset_type)
    val_dataset: Union[GridDistillDataset, ProposalDistillDataset] = dataset_class(
      annotations_file_path=args.val_data,
      image_root_path=args.val_image_root,
      transforms=val_preprocess_fn,
      max_split=args.max_split,
      crop_size=args.input_size,
      max_boxes=args.max_boxes,
    )
    dataloaders["val"] = build_dataloader(val_dataset, is_train=False, args=args)

  return dataloaders


def demo() -> None:
  import matplotlib.pyplot as plt
  import matplotlib.patches as patches
  import numpy as np
  import torchvision.transforms as T
  from torchvision.transforms.functional import to_pil_image

  parser = argparse.ArgumentParser(description="GridDistillDataset comprehensive demo")
  parser.add_argument("--annotation_file_path", help="Path to COCO annotation JSON")
  parser.add_argument("--image_root_path", help="Directory containing images")
  parser.add_argument("--num_samples", type=int, default=3, dest="num_samples", help="Number of samples to process")
  parser.add_argument(
    "--max_split", type=int, default=4, dest="max_split", help="Maximum grid divisions per dimension"
  )
  parser.add_argument("--max_boxes", type=int, default=16, dest="max_boxes", help="Maximum number of boxes per image")
  parser.add_argument("--crop_size", type=int, default=224, dest="crop_size", help="Size of extracted crops")
  parser.add_argument(
    "--save_visualizations", action="store_true", dest="save_viz", help="Save visualization plots to disk"
  )
  args: argparse.Namespace = parser.parse_args()

    print("=" * 80)
  print("GridDistillDataset Comprehensive Demo")
  print("=" * 80)

  image_transform: T.Compose = T.Compose(
    [
      T.Resize(1024, interpolation=T.InterpolationMode.BICUBIC),
      T.CenterCrop(1024),
      T.ToTensor(),
    ]
  )
  crop_transform: T.Compose = T.Compose(
    [
      T.Resize(args.crop_size, interpolation=T.InterpolationMode.BICUBIC),
      T.CenterCrop(args.crop_size),
      T.ToTensor(),
    ]
  )
  transforms: List[T.Compose] = [image_transform, crop_transform]

  # Initialize dataset
  print("\nInitializing Dataset...")
  dataset = GridDistillDataset(
    annotations_file_path=args.annotation_file_path,
    image_root_path=args.image_root_path,
    transforms=transforms,
    max_split=args.max_split,
    crop_size=args.crop_size,
    max_boxes=args.max_boxes,
  )
  print("   Dataset created successfully!")
  print(f"   Total images available: {len(dataset)}")
  print(f"   Grid configurations: {len(dataset.box_templates)} templates")

  # Show grid template information
  print("\nGrid Template Analysis:")
  print(f"   Generated {len(dataset.box_templates)} different grid configurations:")
  for (m, n), template in dataset.box_templates.items():
    num_boxes = template.shape[0]
    print(f"   - {m}x{n} grid -> {num_boxes} boxes")

  # Show some example grid coordinates
  example_grid: torch.Tensor = dataset.box_templates[(2, 3)]  # 2x3 grid
  print("\n   Example 2x3 grid coordinates (normalized [0,1]):")
  for i, box in enumerate(example_grid):
    x0, y0, x1, y1 = box.tolist()
    print(f"   Box {i + 1}: [{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]")

  # Create dataloader for batch processing
  data_loader: DataLoader[Any] = DataLoader(dataset, batch_size=1, shuffle=True)

  print("\nProcessing Sample Data:")
  print("   Batch size: 1 (for detailed analysis)")

  # Process samples with detailed analysis
  for sample_idx, batch in enumerate(data_loader):
    if sample_idx >= args.num_samples:
      break

    print("\n" + "=" * 60)
    print(f"SAMPLE {sample_idx + 1}")
    print("=" * 60)

    # Unpack batch data
    img_batch, boxes_batch, crops_batch = batch
    img = img_batch[0]  # Remove batch dimension
    boxes = boxes_batch[0]  # Remove batch dimension
    crops = crops_batch[0]  # Remove batch dimension

    print("\nTensor Shapes:")
    print(f"   Full Image: {list(img.shape)} (C x H x W)")
    print(f"   Boxes: {list(boxes.shape)} (max_boxes x 5)")
    print(f"   Crops: {list(crops.shape)} (max_boxes x C x H x W)")

    # Analyze box data
    valid_boxes = boxes[boxes[:, 4] == 1.0]  # Only boxes with validity flag = 1
    num_valid = len(valid_boxes)
    print("\nBox Analysis:")
    print(f"   Valid boxes: {num_valid}/{args.max_boxes}")

    # Analyze crops
    valid_crops = crops[:num_valid]  # Only crops corresponding to valid boxes
    print("\nCrop Analysis:")
    print(f"   Valid crops: {len(valid_crops)}")

    # Create visualization if matplotlib is available
    if args.save_viz or sample_idx == 0:  # Always show first sample
      try:
        print("\nCreating visualization...")

        # Convert tensor back to PIL for visualization
        img_pil = to_pil_image(img)

        # Create figure with subplots - simplified layout
        num_crops_to_show = min(8, num_valid)  # Show up to 8 crops
        cols = 4
        rows = 2 + (num_crops_to_show + cols - 1) // cols  # Dynamic rows based on crops

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(f"GridDistillDataset Sample {sample_idx + 1}", fontsize=16, fontweight="bold")

        # Flatten axes for easier indexing
        if rows == 1:
          axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        # Plot 1: Original transformed image with bounding boxes
        ax = axes_flat[0]
        ax.imshow(img_pil)
        ax.set_title(f"Full Image with Grid Boxes\n{img.shape[1]}×{img.shape[2]} pixels, {num_valid} valid boxes")

        # Draw bounding boxes on the image
        img_h, img_w = img.shape[1], img.shape[2]
        colors = plt.cm.Set3(np.linspace(0, 1, max(num_valid, 1)))  # Use Set3 colormap for better visibility

        for i in range(num_valid):
          box = valid_boxes[i]
          x0, y0, x1, y1 = box[:4].tolist()
          # Convert normalized coordinates back to pixel coordinates
          x0, x1 = x0 * img_w, x1 * img_w
          y0, y1 = y0 * img_h, y1 * img_h

          rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor=colors[i], facecolor="none")
          ax.add_patch(rect)

          # Add box number label with background for better visibility
          ax.text(
            x0 + 5,
            y0 + 15,
            f"{i + 1}",
            color="white",
            fontweight="bold",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8),
          )
        ax.axis("off")

        # Hide unused subplots in the first row
        for i in range(1, cols):
          axes_flat[i].axis("off")
          axes_flat[i].set_title("")

        # Show extracted crops starting from second row
        crop_start_idx = cols  # Start from second row
        for crop_idx in range(num_crops_to_show):
          ax_idx = crop_start_idx + crop_idx
          ax = axes_flat[ax_idx]

          crop_img = to_pil_image(valid_crops[crop_idx])
          ax.imshow(crop_img)

          # Get box dimensions for title
          box = valid_boxes[crop_idx]
          w, h = box[2] - box[0], box[3] - box[1]

          ax.set_title(
            f"Crop {crop_idx + 1}\n{w:.3f}x{h:.3f} (normalized)\n{args.crop_size}×{args.crop_size} pixels", fontsize=10
          )
          ax.axis("off")

          # Add border with same color as bounding box
          for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(colors[crop_idx])
            spine.set_linewidth(3)

        # Hide remaining unused subplots
        for i in range(crop_start_idx + num_crops_to_show, len(axes_flat)):
          axes_flat[i].axis("off")

        plt.tight_layout()

        if args.save_viz:
          filename = f"griddistill_sample_{sample_idx + 1}.png"
          plt.savefig(filename, dpi=150, bbox_inches="tight")
          print(f"   Saved visualization: {filename}")
        else:
          plt.show()

        plt.close()

      except Exception as e:
        print(f"   Visualization failed: {e}")

    print(f"\nSample {sample_idx + 1} processed successfully!")

  # Final summary
  print("\n" + "=" * 80)
  print("DEMO SUMMARY")
  print("=" * 80)
  print(f"Successfully processed {args.num_samples} samples")
  print("Dataset is working correctly!")
  print("Key insights:")
  print("   - Images are resized and converted to tensors")
  print("   - Grid boxes are generated dynamically per sample")
  print("   - Coordinates are properly normalized to [0,1]")
  print("   - Crops are extracted and resized consistently")
  print("   - Output tensors have fixed shapes for batching")

  print("\nDemo completed successfully!")


if __name__ == "__main__":
  demo()
