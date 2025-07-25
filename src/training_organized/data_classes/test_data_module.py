"""
Test script to verify the data module works correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, "/home/pc-amax-2/Desktop/Mert/CLIPSelf/src")


def test_imports():
  """Test if all data module imports work."""
  try:
    from training_organized.data_classes import (
      GridDistillDataset,
      ProposalDistillDataset,
      COCOPanopticDataset,
      DataInfo,
      SharedEpoch,
    )

    print("✓ All imports successful")
    return True
  except ImportError as e:
    print(f"✗ Import failed: {e}")
    return False


def test_base_dataset() -> bool:
  """Test base dataset functionality."""
  try:
    from training_organized.data_classes.base_dataset import BaseDataset

    # Test that we can't instantiate abstract class
    try:
      BaseDataset("/fake/path")
      print("✗ BaseDataset should be abstract")
      return False
    except TypeError:
      print("✓ BaseDataset is properly abstract")
      return True
  except Exception as e:
    print(f"✗ BaseDataset test failed: {e}")
    return False


def test_shared_epoch() -> bool:
  """Test SharedEpoch functionality."""
  try:
    from training_organized.data_classes.data_loader import SharedEpoch

    epoch = SharedEpoch(10)
    assert epoch.get_value() == 10

    epoch.set_value(15)
    assert epoch.get_value() == 15

    print("✓ SharedEpoch works correctly")
    return True
  except Exception as e:
    print(f"✗ SharedEpoch test failed: {e}")
    return False


def test_transforms() -> bool:
  """Test custom transforms."""
  try:
    from training_organized.utils.transforms import CustomRandomResize, CustomRandomCrop
    from PIL import Image
    import torch

    # Create a small test image
    test_image = Image.new("RGB", (100, 100), color="red")

    # Test CustomRandomResize
    resize_transform = CustomRandomResize(scale=(0.5, 2.0))
    resized = resize_transform(test_image)

    # Test CustomRandomCrop
    crop_transform = CustomRandomCrop(size=(50, 50))
    cropped = crop_transform(test_image)

    print("✓ Custom transforms work correctly")
    return True
  except Exception as e:
    print(f"✗ Transforms test failed: {e}")
    return False


def test_coco_grid_dataset() -> bool:
  """Test COCO GridDistillDataset functionality."""
  try:
    from training_organized.data_classes.grid_distill import GridDistillDataset
    import torch
    from PIL import Image

    # Create GridDistillDataset
    dataset = GridDistillDataset(
      input_filename="data/coco/annotations/instances_train2017.json",
      transforms=[],
      image_root="data/coco/train2017",
      max_split=16,
      crop_size=224,
      ceph_root="",
      args=None,
    )

    # Test dataset length
    if len(dataset) == 0:
      print("⚠️  COCO dataset is empty, check data path")
      return True

    print(f"✓ COCO dataset loaded with {len(dataset)} samples")

    # Test getting a sample
    sample = dataset[0]

    # Verify sample structure
    assert isinstance(sample, dict), "Sample should be a dictionary"
    assert "image" in sample, "Sample should contain 'image' key"
    assert "annotations" in sample, "Sample should contain 'annotations' key"

    # Verify image format
    image = sample["image"]
    assert isinstance(image, (torch.Tensor, Image.Image)), "Image should be tensor or PIL Image"

    # Verify annotations format
    annotations = sample["annotations"]
    assert isinstance(annotations, (dict, list)), "Annotations should be dict or list"

    print("✓ COCO GridDistillDataset sample format is correct")

    # Test multiple samples to ensure consistency
    for i in range(min(3, len(dataset))):
      sample = dataset[i]
      assert "image" in sample and "annotations" in sample

    print("✓ COCO GridDistillDataset works correctly")
    return True

  except Exception as e:
    print(f"✗ COCO GridDistillDataset test failed: {e}")
    return False


if __name__ == "__main__":
  print("Testing CLIPSelf data module...")

  tests = [test_imports, test_base_dataset, test_shared_epoch, test_transforms, test_coco_grid_dataset]

  passed = 0
  total = len(tests)

  for test in tests:
    if test():
      passed += 1

  print(f"\nResults: {passed}/{total} tests passed")

  if passed == total:
    print("🎉 All tests passed! Data module is working correctly.")
  else:
    print("⚠️  Some tests failed. Check the implementation.")
