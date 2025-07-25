"""
Training configuration for CLIPSelf.
"""

from dataclasses import dataclass
from typing import List, Optional

from training_organized.config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
  """Configuration for training settings."""

  # Training basics
  epochs: int = 32
  batch_size: int = 128
  lr: float = 5e-4
  wd: float = 0.2  # weight decay

  # Optimizer settings
  beta1: float = 0.9
  beta2: float = 0.98
  eps: float = 1e-6

  # Learning rate scheduling
  lr_scheduler: str = "cosine"  # cosine, const, const-cooldown
  warmup: int = 2000
  epochs_cooldown: Optional[int] = None
  lr_cooldown_power: float = 1.0
  lr_cooldown_end: float = 0.0
  skip_scheduler: bool = False

  # Training behavior
  accum_freq: int = 1
  seed: int = 0
  workers: int = 4
  grad_clip_norm: Optional[float] = None
  log_every_n_steps: int = 100

  # Checkpointing
  save_frequency: int = 1
  save_most_recent: bool = True
  delete_previous_checkpoint: bool = False
  resume: Optional[str] = None

  # Evaluation
  zeroshot_frequency: int = 1

  # Distributed training
  distributed: bool = False
  use_bn_sync: bool = False
  ddp_static_graph: bool = False
  horovod: bool = False
  gather_with_grad: bool = False
  no_set_device_rank: bool = False
  dist_url: str = "env://"
  dist_backend: str = "nccl"

  # Logging and output
  logs: str = "./logs"
  name: Optional[str] = None
  log_local: bool = False
  debug: bool = False
  copy_codebase: bool = False

  def validate(self) -> None:
    """Validate training configuration."""
    super().validate()

    if self.epochs <= 0:
      raise ValueError("epochs must be positive")

    if self.batch_size <= 0:
      raise ValueError("batch_size must be positive")

    if self.lr <= 0:
      raise ValueError("lr must be positive")

    if self.wd < 0:
      raise ValueError("wd must be non-negative")

    valid_schedulers: List[str] = ["cosine", "const", "const-cooldown"]
    if self.lr_scheduler not in valid_schedulers:
      raise ValueError(f"lr_scheduler must be one of {valid_schedulers}")

    if self.lr_scheduler == "const-cooldown" and self.epochs_cooldown is None:
      raise ValueError("epochs_cooldown must be specified for const-cooldown scheduler")

    if self.accum_freq != 1:
      raise ValueError("accum_freq must be 1 (gradient accumulation not supported)")

    if self.workers < 0:
      raise ValueError("workers must be non-negative")
