class RunningAverageTracker:
  """
  Tracks running statistics for a single scalar metric during training.

  This class maintains both the current value and running average of a metric,
  which is useful for monitoring training progress and logging.
  """

  def __init__(self) -> None:
    """Initialize the tracker with zero values."""
    self.reset()

  def reset(self) -> None:
    """Reset all tracked values to zero."""
    self.current_value: float = 0.0
    self.running_average: float = 0.0
    self.cumulative_sum: float = 0.0
    self.total_count: int = 0

  def update(self, value: float, batch_size: int = 1) -> None:
    """
    Update the tracker with a new value.

    Args:
      value: The new scalar value to track
      batch_size: Number of samples this value represents (for proper averaging)
    """
    self.current_value = value
    self.cumulative_sum += value * batch_size
    self.total_count += batch_size
    self.running_average = self.cumulative_sum / self.total_count
