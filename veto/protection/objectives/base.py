from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor

from veto.protection.wrappers.base import DiTWrapper


class ProtectionObjective(ABC):
    """Interface for protection optimization (loss + reference)."""

    wrapper: DiTWrapper = None
    x_source: Tensor = None

    def set_source(self, x_source: Tensor) -> None:
        self.x_source = x_source

    @abstractmethod
    def loss(self, x_protected: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError
