from typing import List, Optional

import numpy as np

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.aggregation.base_models import BaseAggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat


def compute_numeric_weighted_geometric_mean(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    weights = np.array(weights)
    values = np.array(values)
    exponents = weights / np.sum(weights)
    result = np.prod(values**exponents)
    return float(result)


def compute_ufloat_weighted_geometric_mean(
    values: List[UFloat], weights: Optional[List[float]] = None
) -> UFloat:
    weights = np.array(weights)
    values = np.array(values)
    exponents = weights / np.sum(weights)
    result = np.prod(values**exponents)
    return result  # type: ignore


class WeightedGeometricMeanAggregation(BaseAggregation):
    """
    Computes the weighted geometric mean of a set of values with corresponding weights.

    .. math::

        A = \\left(\\prod_{i=1}^{n} x_i^{w_i} \\right)^{\\frac{1}{\\sum_{i=1}^{n} w_i}}

    Where:
        - :math:`A` is the weighted arithmetic mean
        - :math:`x_i` is each value in the values array
        - :math:`w_i` is the weight corresponding to each value :math:`x_i`
        - :math:`n` is the number of elements in values and weights arrays


    Usage Example:


    >>> from pumas.aggregation import aggregation_catalogue

    >>> aggregator_class = aggregation_catalogue.get("geometric_mean")

    >>> aggregator = aggregator_class()

    >>> values = [1.0, 2.0, 3.0]
    >>> weights = [0.2, 0.3, 0.5]
    >>> result = aggregator.compute_numeric(values=values, weights=weights)
    >>> print(f"{result:.2f}")
    2.13

    >>> result = aggregator(values=values, weights=weights) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    2.13

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> values = [ufloat(1.0, 0.1), ufloat(2.0, 0.2), ufloat(3.0, 0.3)]
    >>> weights = [0.2, 0.3, 0.5]
    >>> result = aggregator.compute_ufloat(values=values, weights=weights)
    >>> print(result)
    2.13+/-0.13
    """

    def compute_numeric(
        self, values: List[float], weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute the weighted geometric mean for numeric input values.

        Args:
            values (List[float]): The list of numeric values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            float: The computed weighted geometric mean.

        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_numeric_weighted_geometric_mean(
            values=new_values, weights=new_weights
        )

    def compute_ufloat(
        self, values: List[UFloat], weights: Optional[List[float]] = None
    ) -> UFloat:
        """
        Compute the weighted geometric mean for uncertain float input values.

        Args:
            values (List[UFloat]): The list of uncertain float values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            UFloat: The computed weighted geometric mean with uncertainty.
        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_ufloat_weighted_geometric_mean(
            values=new_values, weights=new_weights
        )

    __call__ = compute_numeric
