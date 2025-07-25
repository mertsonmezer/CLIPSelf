"""
Training infrastructure for CLIPSelf.
"""

from .trainer import CLIPSelfTrainer
from .loss import CLIPSelfLoss
from .evaluation import evaluate_model

__all__ = ["CLIPSelfTrainer", "CLIPSelfLoss", "evaluate_model"]
