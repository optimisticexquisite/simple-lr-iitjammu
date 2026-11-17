"""Settings that travel together live together."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Everything you might want to change between two training runs."""

    lr: float = 0.01        # learning rate: the size of each step
    epochs: int = 3000      # how many steps to take

    def __post_init__(self):
        # Catch nonsense settings here, at the door, with a clear message -
        # not three files later with a confusing one.
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
