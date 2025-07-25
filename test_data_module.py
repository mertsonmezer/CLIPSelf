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


def test_base_dataset():
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


def test_shared_epoch():
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


def test_transforms():
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


if __name__ == "__main__":
  print("Testing CLIPSelf data module...")

  tests = [test_imports, test_base_dataset, test_shared_epoch, test_transforms]

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
