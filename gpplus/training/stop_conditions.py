from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict


class StopConditionContext(TypedDict):
    """Context passed to stop conditions for evaluation."""

    epoch: int
    model: Any
    trainer: Any
    loss: float
    previous_loss: Optional[float]
    best_loss: float
    no_improvement_epochs: int
    device: str


class StopCondition(ABC):
    """
    Base class for training stop conditions.

    Subclasses should implement the `should_stop` method which returns
    a tuple of (bool, str) indicating whether training should stop and
    the reason for stopping.
    """

    @abstractmethod
    def should_stop(self, context: StopConditionContext) -> tuple[bool, str]:
        """
        Determine if training should stop based on the current context.

        Args:
            context: Dictionary containing training state information.

        Returns:
            Tuple of (should_stop: bool, reason: str)
            - should_stop: True if training should stop, False otherwise
            - reason: Human-readable reason for stopping (empty string if not stopping)
        """
        pass


class ConvergencePatienceStopCondition(StopCondition):
    """
    Stop condition based on patience (no improvement for N epochs).

    Args:
        patience: Number of epochs without improvement before stopping.
                 If None, this condition is disabled.
    """

    def __init__(self, patience: Optional[int]):
        self.patience = patience

    def should_stop(self, context: StopConditionContext) -> tuple[bool, str]:
        if self.patience is None:
            return False, ""

        if context["no_improvement_epochs"] >= self.patience:
            return True, f"No improvement for {self.patience} epochs"

        return False, ""


class MinLossChangeStopCondition(StopCondition):
    """
    Stop condition based on minimum loss change threshold.

    Args:
        min_loss_change: Minimum absolute loss change required to continue training.
                         If the loss change is below this threshold, training stops.
    """

    def __init__(self, min_loss_change: float):
        self.min_loss_change = min_loss_change

    def should_stop(self, context: StopConditionContext) -> tuple[bool, str]:
        previous_loss = context["previous_loss"]
        if previous_loss is None:
            return False, ""

        loss_change = abs(previous_loss - context["loss"])
        if loss_change < self.min_loss_change:
            return True, f"Absolute loss change below {self.min_loss_change:.1e}"

        return False, ""
