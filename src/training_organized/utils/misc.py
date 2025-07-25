"""
Miscellaneous utilities for CLIPSelf training.

This module provides various utility functions used across the codebase.
"""

import numpy as np
from functools import partial
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
  """
  Apply function to a list of arguments.

  Note:
      This function applies the ``func`` to multiple inputs and
      map the multiple outputs of the ``func`` into different
      list. Each list contains the same type of outputs corresponding
      to different inputs.

  Args:
      func: A function that will be applied to a list of arguments

  Returns:
      tuple(list): A tuple containing multiple list, each list contains
          a kind of returned results by the function
  """
  pfunc = partial(func, **kwargs) if kwargs else func
  map_results = map(pfunc, *args)
  return tuple(map(list, zip(*map_results)))


def mask2box(mask: np.ndarray) -> tuple:
  """
  Convert binary mask to bounding box coordinates.

  Args:
      mask: Binary mask as numpy array

  Returns:
      Tuple of (x0, y0, x1, y1) coordinates
  """
  ys, xs = np.where(mask)
  if len(ys) == 0 or len(xs) == 0:
    return 0, 0, 1, 1  # Return minimal box if mask is empty

  y0, y1 = ys.min(), ys.max()
  x0, x1 = xs.min(), xs.max()

  return x0, y0, x1, y1
