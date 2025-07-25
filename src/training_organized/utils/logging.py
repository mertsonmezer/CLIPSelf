"""
Logging utilities for CLIPSelf training.
"""

import logging
import os
from typing import Optional


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO, include_host: bool = False) -> None:
  """
  Set up logging configuration - exactly like original training/logger.py
  """
  if include_host:
    import socket

    hostname = socket.gethostname()
    formatter = logging.Formatter(
      f"%(asctime)s |  {hostname} | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
    )
  else:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S")

  logging.root.setLevel(level)
  loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
  for logger in loggers:
    logger.setLevel(level)

  stream_handler = logging.StreamHandler()
  stream_handler.setFormatter(formatter)
  logging.root.addHandler(stream_handler)

  if log_file:
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
  """
  Get a logger with the specified name.

  Args:
      name: Logger name

  Returns:
      Logger instance
  """
  return logging.getLogger(name)


class LoggerMixin:
  """Mixin class to add logging capabilities to any class."""

  @property
  def logger(self) -> logging.Logger:
    """Get logger for this class."""
    return logging.getLogger(self.__class__.__name__)
