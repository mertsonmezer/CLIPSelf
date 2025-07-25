"""
Data loading utilities for CLIPSelf training.

This module provides dataset factory functions and data loading utilities.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from multiprocessing import Value
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training_organized.data_classes.coco_datasets import COCOPanopticDataset
from training_organized.data_classes.grid_distill import GridDistillDataset
from training_organized.data_classes.proposal_distill import ProposalDistillDataset


class SharedEpoch:
  """Shared epoch counter for distributed training."""

  def __init__(self, epoch: int = 0):
    """
    Initialize shared epoch counter.

    Args:
      epoch (int): Initial epoch value
    """
    self.shared_epoch = Value("i", epoch)

  def set_value(self, epoch: int) -> None:
    """Set epoch value."""
    self.shared_epoch.value = epoch

  def get_value(self) -> int:
    """Get current epoch value."""
    return self.shared_epoch.value


@dataclass
class DataInfo:
  """Container for dataset information including dataloader and sampler."""

  dataloader: DataLoader[Tuple[torch.Tensor, ...]]
  sampler: Optional[DistributedSampler[Tuple[torch.Tensor, ...]]] = None
  shared_epoch: Optional[SharedEpoch] = None

  def set_epoch(self, epoch: int) -> None:
    """
    Set epoch for both shared epoch and distributed sampler.

    Args:
      epoch (int): Epoch number to set
    """
    if self.shared_epoch is not None:
      self.shared_epoch.set_value(epoch)
    if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
      self.sampler.set_epoch(epoch)


def get_grid_distill_dataset(
  args: argparse.Namespace,
  preprocess_fn: List[Callable[..., Any]],
  is_train: bool,
  epoch: int = 0,
  tokenizer=None,  # TODO: Handle tokenizer if needed
) -> DataInfo:
  """
  Create GridDistillDataset with dataloader.

  Args:
    args (argparse.Namespace): Training arguments
    preprocess_fn (List[Callable[..., Any]]): Preprocessing transforms
    is_train (bool): Whether this is training dataset
    epoch (int): Current epoch
    tokenizer : Text tokenizer (unused)

  Returns:
      DataInfo object with dataloader and sampler
  """
  assert is_train, "GridDistillDataset only supports training mode"

  input_filename: str = args.train_data
  assert input_filename, "train_data must be specified"

  logging.info(f"Creating GridDistillDataset from {input_filename}")

  dataset = GridDistillDataset(
    input_filename=input_filename,
    transforms=preprocess_fn,
    image_root=args.train_image_root,
    crop_size=args.input_size,
    max_split=args.max_split,
    ceph_root=getattr(args, "train_ceph_root", ""),
    pre_transforms=getattr(args, "pre_transforms", False),
    args=args,
  )

  num_samples: int = len(dataset)
  logging.info(f"GridDistillDataset created with {num_samples} samples")

  # Create sampler
  sampler: DistributedSampler[Any] | None = DistributedSampler(dataset) if args.distributed else None
  shuffle: bool = is_train and sampler is None

  # Create dataloader
  dataloader: DataLoader[Tuple[torch.Tensor, ...]] = DataLoader(
    dataset,
    batch_size=args.batch_size,
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
  args: argparse.Namespace,
  preprocess_fn: List[Callable[..., Any]],
  is_train: bool,
  epoch: int = 0,
  tokenizer=None,  # TODO: Handle tokenizer if needed
) -> DataInfo:
  """
  Create ProposalDistillDataset with dataloader.

  Args:
    args (argparse.Namespace): Training arguments
    preprocess_fn (List[Callable[..., Any]]): Preprocessing transforms
    is_train (bool): Whether this is training dataset
    epoch (int): Current epoch
    tokenizer: Text tokenizer

  Returns:
    DataInfo object with dataloader and sampler
  """
  assert is_train, "ProposalDistillDataset only supports training mode"

  input_filename: str = args.train_data
  assert input_filename, "train_data must be specified"

  logging.info(f"Creating ProposalDistillDataset from {input_filename}")

  dataset = ProposalDistillDataset(
    input_filename,
    preprocess_fn,
    image_root=args.train_image_root,
    tokenizer=tokenizer,
    crop_size=args.input_size,
    args=args,
  )

  num_samples: int = len(dataset)
  logging.info(f"ProposalDistillDataset created with {num_samples} samples")

  # Create sampler
  sampler: DistributedSampler[Any] | None = DistributedSampler(dataset) if args.distributed else None
  shuffle: bool = is_train and sampler is None

  # Create dataloader
  dataloader: DataLoader[Tuple[torch.Tensor, ...]] = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=shuffle,
    num_workers=args.workers,
    pin_memory=True,
    sampler=sampler,
    drop_last=is_train,
  )

  dataloader.num_samples = num_samples
  dataloader.num_batches = len(dataloader)

  return DataInfo(dataloader, sampler)


def get_coco_panoptic_dataset(
  args: argparse.Namespace,
  preprocess_fn: List[Callable[..., Any]],
  is_train: bool,
  epoch: int = 0,
  tokenizer=None,  # TODO: Handle tokenizer if needed
) -> DataInfo:
  """
  Create COCOPanopticDataset with dataloader.

  Args:
    args (argparse.Namespace): Training arguments
    preprocess_fn (List[Callable[..., Any]]): Preprocessing transforms
    is_train (bool): Whether this is training dataset
    epoch (int): Current epoch
    tokenizer: Text tokenizer

  Returns:
    DataInfo object with dataloader and sampler
  """
  input_filename: str = args.train_data if is_train else args.val_data
  assert input_filename, f"{'train_data' if is_train else 'val_data'} must be specified"

  logging.info(f"Creating COCOPanopticDataset from {input_filename}")

  dataset = COCOPanopticDataset(
    input_filename,
    preprocess_fn,
    segm_root=args.val_segm_root,
    image_root=args.val_image_root,
    embed_path=args.embed_path,
    tokenizer=tokenizer,
    crop_size=args.input_size,
    min_size=getattr(args, "min_size", 8),
    max_size=getattr(args, "max_size", 1024),
    downsample_factor=getattr(args, "downsample_factor", 16),
  )

  num_samples: int = len(dataset)
  logging.info(f"COCOPanopticDataset created with {num_samples} samples")

  # Create sampler
  sampler: DistributedSampler[Any] | None = DistributedSampler(dataset) if args.distributed else None
  shuffle: bool = is_train and sampler is None

  # Batch size handling
  batch_size: int = args.batch_size if is_train else min(args.batch_size, 1)

  # Create dataloader
  dataloader: DataLoader[Tuple[torch.Tensor, ...]] = DataLoader(
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


def get_dataset_fn(data_path: str, dataset_type: str) -> Callable[..., DataInfo]:
  """
  Get dataset factory function based on dataset type.

  Args:
    data_path (str): Path to dataset
    dataset_type (str): Type of dataset

  Returns:
    Dataset factory function

  Raises:
    ValueError: If dataset type is not supported
  """
  dataset_map: Dict[str, Callable[..., DataInfo]] = {
    "coco_panoptic": get_coco_panoptic_dataset,
    "proposals_distill": get_proposal_distill_dataset,
    "grid_distill": get_grid_distill_dataset,
  }

  if dataset_type not in dataset_map:
    raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: {list(dataset_map.keys())}")

  return dataset_map[dataset_type]


def get_data(
  args: argparse.Namespace,
  preprocess_fns: List[Tuple[Callable[..., Any]]],
  epoch: int = 0,
  tokenizer=None,  # TODO: Handle tokenizer if needed
) -> Dict[str, Callable[..., DataInfo]]:
  """
  Create training and validation datasets.

  Args:
    args (argparse.Namespace): Training arguments
    preprocess_fns (List[Tuple[Callable[..., Any]]]): Tuple of (train_preprocess, val_preprocess)
    epoch (int): Current epoch
    tokenizer: Text tokenizer

  Returns:
    Dictionary with 'train' and/or 'val' DataInfo objects
  """
  preprocess_train, preprocess_val = preprocess_fns
  data: Dict[str, Callable[..., DataInfo]] = {}

  # Create training dataset
  if args.train_data:
    logging.info("Creating training dataset")
    data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
      args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
    )

  # Create validation dataset
  if args.val_data:
    logging.info("Creating validation dataset")
    test_type: str = getattr(args, "test_type", args.dataset_type)
    data["val"] = get_dataset_fn(args.val_data, test_type)(args, preprocess_val, is_train=False, tokenizer=tokenizer)

  return data
