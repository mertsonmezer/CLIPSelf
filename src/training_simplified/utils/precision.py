from contextlib import suppress
from typing import Any, Callable, Union

import torch


def get_autocast(precision: str) -> Union[Callable[..., Any], Any]:
  """
  Get the appropriate autocast context manager based on precision setting.

  This function returns the correct autocast context for mixed precision training,
  which helps reduce memory usage and can speed up training on compatible hardware.

  Args:
    precision (str): The precision mode to use. Supported values:
      - 'amp': Standard automatic mixed precision with float16
      - 'amp_bfloat16' or 'amp_bf16': Mixed precision with bfloat16 (more stable)
      - Any other value: No mixed precision (full precision)

  Returns:
    Union[Callable[..., Any], Any]:
      - torch.cuda.amp.autocast for 'amp' mode
      - Lambda function returning bfloat16 autocast for 'amp_bfloat16'/'amp_bf16'
      - suppress context manager for full precision (no-op)
  """
  if precision == "amp":
    # Standard mixed precision with float16
    return torch.cuda.amp.autocast

  elif precision in ("amp_bfloat16", "amp_bf16"):
    # Mixed precision with bfloat16 - more stable than float16 for CLIP training
    # bfloat16 has better numerical stability due to larger exponent range
    return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)

  else:
    return suppress
