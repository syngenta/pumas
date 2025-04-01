from abc import abstractmethod
from typing import List, Optional, Union

from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat


class Aggregation(AbstractParametrizedStrategy):
    @abstractmethod
    def compute_numeric(
        self,
        values: List[Union[float, None]],
        weights: Optional[List[Union[float, None]]],
    ) -> float:
        """Computes the aggregation of numeric values with corresponding weights."""

    @abstractmethod
    def compute_ufloat(
        self,
        values: List[Union[UFloat, None]],
        weights: Optional[List[Union[float, None]]],
    ) -> UFloat:
        """Computes the aggregation of ufloat values with corresponding weights."""
