from typing import List, Optional, Union

import numpy as np

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.aggregation.base_models import Aggregation
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat


def compute_numeric_weighted_summation(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    values, weights = run_data_validation_pipeline(values=values, weights=weights)
    result: float = np.sum(np.multiply(values, weights))
    return result


def compute_ufloat_weighted_summation(
    values: List[UFloat], weights: Optional[List[float]] = None
) -> UFloat:
    values, weights = run_data_validation_pipeline(values=values, weights=weights)
    result: UFloat = np.sum(np.multiply(values, weights))
    return result


class WeightedSummationAggregation(Aggregation):
    """
     Computes the weighted summation of a set of values with corresponding weights.

     .. math::

         A = \\sum_{i=1}^{n}{w_i x_i}

     Where:
         - :math:`A` is the weighted summation
         - :math:`x_i` is each value in the values array
         - :math:`w_i` is the weight corresponding to each value :math:`x_i`
         - :math:`n` is the number of elements in the values and weights arrays

    Usage Example:

     >>> from pumas.aggregation import aggregation_catalogue

     >>> aggregator_class = aggregation_catalogue.get("summation")

     >>> aggregator = aggregator_class()

     >>> values = [1.0, 2.0, 3.0]
     >>> weights = [0.2, 0.3, 0.5]
     >>> result = aggregator.compute_numeric(values=values, weights=weights)
     >>> print(f"{result:.2f}")
     2.30

     >>> result = aggregator(values=values, weights=weights) # Same as compute_numeric
     >>> print(f"{result:.2f}")
     2.30

     >>> from uncertainties import ufloat
     >>> values = [ufloat(1.0, 0.1), ufloat(2.0, 0.2), ufloat(3.0, 0.3)]
     >>> weights = [0.2, 0.3, 0.5]
     >>> result = aggregator.compute_ufloat(values=values, weights=weights)
     >>> print(result)
     2.30+/-0.16
    """

    def compute_numeric(
        self,
        values: List[Union[float, None]],
        weights: Optional[List[Union[float, None]]] = None,
    ) -> float:
        """
        Compute the weighted summation for numeric input values.

        Args:
            values (List[float]): The list of numeric values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            float: The computed weighted summation.

        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_numeric_weighted_summation(
            values=new_values, weights=new_weights
        )

    def compute_ufloat(
        self,
        values: List[Union[UFloat, None]],
        weights: Optional[List[Union[float, None]]] = None,
    ) -> UFloat:
        """
        Compute the weighted summation for uncertain float input values.

        Args:
            values (List[UFloat]): The list of uncertain float values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            UFloat: The computed weighted summation with uncertainty.
        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_ufloat_weighted_summation(values=new_values, weights=new_weights)

    __call__ = compute_numeric
