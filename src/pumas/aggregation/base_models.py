from abc import abstractmethod
from typing import List, Union

from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.uncertainty.uncertainties_wrapper import UFloat


class BaseAggregation(AbstractParametrizedStrategy):
    @abstractmethod
    def compute_numeric(
        self, values: List[float], weights: List[float]
    ) -> Union[float, UFloat]:
        """Computes the aggregation of numeric values with corresponding weights."""

    @abstractmethod
    def compute_ufloat(
        self, values: List[UFloat], weights: List[float]
    ) -> Union[float, UFloat]:
        """Computes the aggregation of ufloat values with corresponding weights."""
