"""
COCO API utilities for CLIPSelf training.

This module provides enhanced COCO API classes with snake case aliases
and panoptic segmentation support.
"""

import warnings

import pycocotools
from pycocotools.coco import COCO as _COCO
from pycocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):
  """
  Enhanced COCO class with snake case function aliases.

  This class is almost the same as official pycocotools package.
  It implements some snake case function aliases so that the COCO class has
  the same interface as LVIS class.
  """

  def __init__(self, annotation_file=None):
    """
    Initialize COCO API.

    Args:
        annotation_file: Path to annotation file
    """
    if getattr(pycocotools, "__version__", "0") >= "12.0.2":
      warnings.warn(
        'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"', UserWarning
      )
    super().__init__(annotation_file=annotation_file)
    self.img_ann_map = self.imgToAnns
    self.cat_img_map = self.catToImgs

  def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
    """Get annotation IDs (snake case alias)."""
    return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

  def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
    """Get category IDs (snake case alias)."""
    return self.getCatIds(cat_names, sup_names, cat_ids)

  def get_img_ids(self, img_ids=[], cat_ids=[]):
    """Get image IDs (snake case alias)."""
    return self.getImgIds(img_ids, cat_ids)

  def load_anns(self, ids):
    """Load annotations (snake case alias)."""
    return self.loadAnns(ids)

  def load_cats(self, ids):
    """Load categories (snake case alias)."""
    return self.loadCats(ids)

  def load_imgs(self, ids):
    """Load images (snake case alias)."""
    return self.loadImgs(ids)


class COCOPanoptic(COCO):
  """
  COCO API for panoptic segmentation.

  Extends the base COCO class to handle panoptic segmentation annotations.
  """

  def __init__(self, annotation_file=None):
    """
    Initialize COCO Panoptic API.

    Args:
        annotation_file: Path to panoptic annotation file
    """
    super().__init__(annotation_file)


# Alias for compatibility
COCOeval = _COCOeval
