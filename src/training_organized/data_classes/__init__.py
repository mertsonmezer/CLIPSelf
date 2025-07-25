"""
Data module for CLIPSelf training.

This module provides clean and organized dataset implementations for CLIPSelf training,
including grid distillation and proposal distillation datasets.
"""

from .grid_distill import GridDistillDataset
from .proposal_distill import ProposalDistillDataset
from .coco_datasets import COCOPanopticDataset
from .data_loader import (
  get_grid_distill_dataset,
  get_proposal_distill_dataset,
  get_coco_panoptic_dataset,
  get_dataset_fn,
  get_data,
  DataInfo,
  SharedEpoch,
)

__all__ = [
  "GridDistillDataset",
  "ProposalDistillDataset",
  "COCOPanopticDataset",
  "get_grid_distill_dataset",
  "get_proposal_distill_dataset",
  "get_coco_panoptic_dataset",
  "get_dataset_fn",
  "get_data",
  "DataInfo",
  "SharedEpoch",
]
