"""
Zero-shot evaluation module for CLIPSelf model.

This module provides functions to evaluate the zero-shot performance of the CLIPSelf model
on different region extraction methods (ROIs, crops, mask pooling) and compute mean
accuracy across classes (mACC) for both "thing" and "stuff" categories.
"""

import argparse
import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from open_clip import get_cast_dtype
from training.utils.dist_utils import all_gather
from training.utils.distributed import is_master
from training.utils.precision import get_autocast


def run(
  model: torch.nn.Module, dataloader: DataLoader[Tuple[torch.Tensor, ...]], args: argparse.Namespace
) -> Tuple[torch.Tensor, ...]:
  """
  Run zero-shot evaluation on the given model and dataloader.

  This function extracts features using three different methods:
  1. ROI features: Features extracted from region proposals
  2. Crop features: Features extracted from image crops
  3. Mask pooling features: Features extracted using mask pooling

  Args:
    model (torch.nn.Module): The CLIPSelf model to evaluate
    dataloader (DataLoader): DataLoader containing validation data with embeddings
    args (argparse.Namespace): Configuration arguments containing:
      - device: Device to run evaluation on
      - precision: Precision for autocast (e.g., 'fp16', 'fp32')
      - distributed: Whether running in distributed mode
      - horovod: Whether using Horovod for distributed training
      - extract_type: Type of feature extraction ('v1' or 'v2')
      - image_ave_pool: Whether to use average pooling for image features

  Returns:
    Tuple containing:
      - correct_rois: Correctness matrix for ROI predictions [N, 5]
      - correct_crops: Correctness matrix for crop predictions [N, 5]
      - correct_maskpool: Correctness matrix for mask pooling predictions [N, 5]
      - similarity_rois: Similarity scores for ROI features [N]
      - similarity_crops: Similarity scores for crop features [N]
      - similarity_maskpool: Similarity scores for mask pooling features [N]
      - all_box_sizes: Box sizes for all predictions [N]
      - all_is_thing: Thing/stuff labels for all predictions [N]
      - all_cls_labels: Class labels for all predictions [N]
  """
  # Get class embeddings from dataset and normalize them
  cls_embeddings = dataloader.dataset.embeddings
  cls_embeddings = F.normalize(torch.from_numpy(cls_embeddings).float(), dim=-1)
  cls_embeddings = cls_embeddings.to(args.device)

  # Setup precision and casting
  autocast = get_autocast(args.precision)
  cast_dtype = get_cast_dtype(args.precision)
  if cast_dtype is not None:
    cls_embeddings = cls_embeddings.to(dtype=cast_dtype)

  # Initialize result containers
  with torch.no_grad():
    # Correctness matrices for top-5 predictions
    correct_rois = []
    correct_maskpool = []
    correct_crops = []

    # Similarity scores for ground truth classes
    similarity_crops = []
    similarity_rois = []
    similarity_maskpool = []

    # Additional metadata
    all_box_sizes = []
    all_is_thing = []
    all_cls_labels = []

    # Process each batch
    for images, bboxes, image_crops, gt_masks, masked_image_crops in tqdm(dataloader, disable=not is_master(args)):
      # Move data to device
      images = images.to(args.device)
      bboxes = bboxes.to(args.device)
      image_crops = image_crops.to(args.device)
      masked_image_crops = masked_image_crops.to(args.device)
      gt_masks = gt_masks.to(args.device)

      # Cast to appropriate dtype if needed
      if cast_dtype is not None:
        images = images.to(dtype=cast_dtype)
        bboxes = bboxes.to(dtype=cast_dtype)
        image_crops = image_crops.to(dtype=cast_dtype)
        masked_image_crops = masked_image_crops.to(dtype=cast_dtype)
        gt_masks = gt_masks.to(dtype=cast_dtype)

      # Process data per image in the batch
      image_crops_list = []
      gt_masks_list = []
      cls_labels = []
      rois = []
      box_sizes = []
      is_thing = []

      for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image in zip(
        bboxes, image_crops, gt_masks, masked_image_crops
      ):
        # Filter valid bounding boxes (confidence > 0.5)
        valid = bboxes_per_image[:, 5] > 0.5

        # Extract valid data
        rois.append(bboxes_per_image[valid, :4])  # x1, y1, x2, y2
        cls_labels.append(bboxes_per_image[valid, 4])  # class labels
        image_crops_list.append(crops_per_image[valid])
        gt_masks_list.append(gt_mask[valid])
        box_sizes.append(bboxes_per_image[valid, 6])  # box sizes
        is_thing.append(bboxes_per_image[valid, 7])  # thing/stuff labels

      # Concatenate all valid data
      cls_labels = torch.cat(cls_labels, dim=0).to(torch.long)

      # Skip batch if no valid boxes
      if cls_labels.shape[0] == 0:
        continue

      image_crops = torch.cat(image_crops_list)
      box_sizes = torch.cat(box_sizes, dim=0).float()
      is_thing = torch.cat(is_thing, dim=0)

      # Store batch metadata
      all_box_sizes.append(box_sizes)
      all_is_thing.append(is_thing)

      # Feature extraction and prediction
      with autocast():
        # Get model module (handle distributed training)
        if args.distributed and not args.horovod:
          module = model.module
        else:
          module = model

        # Extract ROI features using pseudo box encoding
        roi_extractor = module.encode_pseudo_boxes
        roi_features = roi_extractor(images, rois, normalize=True, extract_type=args.extract_type)

        # Extract mask pooling features
        mask_pooler = module.encode_masks
        maskpool_features = mask_pooler(images, gt_masks_list, normalize=True, mask_attn=args.extract_type == "v1")

        # Extract crop features (two different methods)
        if args.image_ave_pool:
          # Method 1: Dense encoding with average pooling
          feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
          crop_features = feature_map.mean(dim=(-2, -1))
          crop_features = F.normalize(crop_features, dim=-1)
        else:
          # Method 2: Direct image encoding
          crop_features = module.encode_image(image_crops, normalize=True)

        # Cast features to appropriate dtype
        if cast_dtype is not None:
          roi_features = roi_features.to(dtype=cast_dtype)
          crop_features = crop_features.to(dtype=cast_dtype)
          maskpool_features = maskpool_features.to(dtype=cast_dtype)

        # Compute similarity logits with class embeddings
        roi_logits = roi_features @ cls_embeddings.T
        crop_logits = crop_features @ cls_embeddings.T
        maskpool_logits = maskpool_features @ cls_embeddings.T

      # Get top-5 predictions for each method
      _, roi_top5_inds = roi_logits.topk(5)
      _, crop_top5_inds = crop_logits.topk(5)
      _, maskpool_top5_inds = maskpool_logits.topk(5)

      # Compute correctness matrices (True if ground truth is in top-k)
      correct_rois.append(roi_top5_inds == cls_labels.view(-1, 1))
      correct_crops.append(crop_top5_inds == cls_labels.view(-1, 1))
      correct_maskpool.append(maskpool_top5_inds == cls_labels.view(-1, 1))

      # Compute similarity scores for ground truth classes
      similarity_rois.append(torch.gather(roi_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
      similarity_crops.append(torch.gather(crop_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])
      similarity_maskpool.append(torch.gather(maskpool_logits, dim=1, index=cls_labels.view(-1, 1))[:, 0])

      all_cls_labels.append(cls_labels)

    # Concatenate all batch results
    correct_rois = torch.cat(correct_rois).float()
    correct_crops = torch.cat(correct_crops).float()
    correct_maskpool = torch.cat(correct_maskpool).float()
    similarity_rois = torch.cat(similarity_rois).float()
    similarity_crops = torch.cat(similarity_crops).float()
    similarity_maskpool = torch.cat(similarity_maskpool).float()
    all_box_sizes = torch.cat(all_box_sizes)
    all_is_thing = torch.cat(all_is_thing)
    all_cls_labels = torch.cat(all_cls_labels)

    # Synchronize results across GPUs in distributed training
    if args.distributed and not args.horovod:
      correct_rois = multi_gpu_sync(correct_rois)
      correct_crops = multi_gpu_sync(correct_crops)
      correct_maskpool = multi_gpu_sync(correct_maskpool)
      all_box_sizes = multi_gpu_sync(all_box_sizes)
      all_is_thing = multi_gpu_sync(all_is_thing)
      similarity_rois = multi_gpu_sync(similarity_rois)
      similarity_crops = multi_gpu_sync(similarity_crops)
      similarity_maskpool = multi_gpu_sync(similarity_maskpool)
      all_cls_labels = multi_gpu_sync(all_cls_labels)

  return (
    correct_rois,
    correct_crops,
    correct_maskpool,
    similarity_rois,
    similarity_crops,
    similarity_maskpool,
    all_box_sizes,
    all_is_thing,
    all_cls_labels,
  )


def multi_gpu_sync(x: torch.Tensor) -> torch.Tensor:
  """
  Synchronize tensor across multiple GPUs in distributed training.

  Args:
      x: Input tensor to synchronize across GPUs

  Returns:
      Concatenated tensor from all GPUs
  """
  device = x.device
  x_list = all_gather(x.cpu())
  x = torch.cat([res.to(device) for res in x_list])
  return x


def macc_with_is_thing(
  correct_matrix: torch.Tensor, is_thing: torch.Tensor, all_cls_labels: torch.Tensor, prefix: str
) -> Dict[str, float]:
  """
  Compute mean accuracy across classes (mACC) separated by thing/stuff categories.

  Args:
      correct_matrix: Boolean matrix indicating correct predictions [N, K] where K is top-k
      is_thing: Binary tensor indicating if each sample is a "thing" (1) or "stuff" (0) [N]
      all_cls_labels: Class labels for all predictions [N]
      prefix: Prefix for result keys (e.g., 'rois', 'crops', 'maskpool')

  Returns:
      Dictionary containing top-1 and top-5 mACC for thing and stuff categories
  """

  def _macc(corrects: torch.Tensor, cls_labels: torch.Tensor) -> float:
    """
    Calculate mean accuracy across classes for given predictions.

    Args:
        corrects: Boolean tensor indicating correct predictions
        cls_labels: Class labels corresponding to predictions

    Returns:
        Mean accuracy across all classes
    """
    min_id = int(cls_labels.min().item())
    max_id = int(cls_labels.max().item())
    cand_labels = list(range(min_id, max_id + 1))

    acc_per_cls = []

    for lb in cand_labels:
      corrects_per_cls = corrects[cls_labels == lb]
      if corrects_per_cls.shape[0] == 0:
        continue
      acc_per_cls.append(corrects_per_cls.mean().half().item())

    return sum(acc_per_cls) / len(acc_per_cls)

  results = {}

  # Separate predictions by thing/stuff categories
  thing_correct_matrix = correct_matrix[is_thing > 0]
  stuff_correct_matrix = correct_matrix[is_thing < 1]

  thing_cls_labels = all_cls_labels[is_thing > 0].long()
  stuff_cls_labels = all_cls_labels[is_thing < 1].long()

  # Calculate accuracies for "thing" categories
  thing_top1_acc = _macc(thing_correct_matrix[:, 0], thing_cls_labels)
  thing_top5_acc = _macc(thing_correct_matrix.sum(-1), thing_cls_labels)

  # Calculate accuracies for "stuff" categories
  stuff_top1_acc = _macc(stuff_correct_matrix[:, 0], stuff_cls_labels)
  stuff_top5_acc = _macc(stuff_correct_matrix.sum(-1), stuff_cls_labels)

  # Store results with descriptive keys
  results[f"{prefix}.thing.macc1"] = thing_top1_acc
  results[f"{prefix}.thing.macc5"] = thing_top5_acc
  results[f"{prefix}.stuff.macc1"] = stuff_top1_acc
  results[f"{prefix}.stuff.macc5"] = stuff_top5_acc

  return results


def zero_shot_eval(model: torch.nn.Module, data: Dict[str, Any], epoch: int, args: Any) -> Dict[str, float]:
  """
  Perform zero-shot evaluation of the model.

  This function evaluates the model's zero-shot performance on validation data
  using different feature extraction methods and computes mean accuracy across classes.

  Args:
      model: The CLIPSelf model to evaluate
      data: Dictionary containing data loaders with 'val' key for validation data
      epoch: Current training epoch
      args: Configuration arguments containing:
          - zeroshot_frequency: How often to run zero-shot evaluation
          - epochs: Total number of training epochs

  Returns:
      Dictionary containing evaluation results for different extraction methods
      and thing/stuff categories. Empty dict if evaluation is skipped.
  """
  # Skip evaluation if no validation data
  if "val" not in data:
    return {}

  # Skip evaluation if frequency is 0 (disabled)
  if args.zeroshot_frequency == 0:
    return {}

  # Skip evaluation if not at the right frequency (except for final epoch)
  if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
    return {}

  logging.info("Running zero-shot region classifier evaluation")

  # Run evaluation and collect results
  results = {}
  correct_rois, correct_crops, correct_maskpool, _, _, _, _, all_is_thing, all_cls_labels = run(
    model, data["val"].dataloader, args
  )

  # Compute mACC for each extraction method
  results.update(macc_with_is_thing(correct_rois, all_is_thing, all_cls_labels, "rois"))
  results.update(macc_with_is_thing(correct_crops, all_is_thing, all_cls_labels, "crops"))
  results.update(macc_with_is_thing(correct_maskpool, all_is_thing, all_cls_labels, "maskpool"))

  return results
