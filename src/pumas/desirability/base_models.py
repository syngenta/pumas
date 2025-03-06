from abc import abstractmethod

from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.uncertainty.uncertainties_wrapper import UFloat


class Desirability(AbstractParametrizedStrategy):
    """Abstract base class for desirability functions."""

    @abstractmethod
    def compute_numeric(self, x: float) -> float:
        """Computes the desirability score on numeric values."""
        pass

    @abstractmethod
    def compute_ufloat(self, x: UFloat) -> UFloat:
        """Computes the desirability score on UFloat values."""
        pass
