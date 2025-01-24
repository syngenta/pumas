from typing import List

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.architecture.parametrized_strategy import AbstractParametrizedStrategy
from pumas.error_propagation.uncertainties import UFloat


class BaseAggregation(AbstractParametrizedStrategy):
    def compute_score(self, values: List[float], weights: List[float] = None) -> float:
        if weights is None:
            weights = [1.0] * len(values)
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )

        result = self._get_partial_utility_function(
            values=new_values, weights=new_weights
        )

        return result

    def compute_uscore(
        self, values: List[UFloat], weights: List[float] = None
    ) -> UFloat:
        if weights is None:
            weights = [1.0] * len(values)

        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )

        result = self._get_partial_utility_function(
            values=new_values, weights=new_weights
        )
        return result
