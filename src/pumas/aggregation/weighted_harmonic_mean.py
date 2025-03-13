from typing import Any, Dict, List, Optional

import numpy as np

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.aggregation.base_models import Aggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat


def weighted_harmonic_mean(values, weights):
    """
    Computes the weighted harmonic mean of a set of values with corresponding weights.

    .. math::

        H = \\frac{\\sum_{i=1}^{n} w_i}{\\sum_{i=1}^{n} \\frac{w_i}{x_i}}

    Where:
        - :math:`H` is the weighted harmonic mean
        - :math:`x_i` are the values
        - :math:`w_i` are the corresponding weights
        - :math:`n` is the number of observations

    Args:
        values (Iterable): The values to be averaged.
        weights (Iterable): The weights for each value.

    Returns:
         Union[float,UFloat]: Resultant weighted harmonic mean.
    """
    weights = np.array(weights)
    values = np.array(values)
    weighted_reciprocal_sum = np.sum(weights / values)
    total_weight = np.sum(weights)
    result = total_weight / weighted_reciprocal_sum
    return result


def compute_numeric_weighted_harmonic_mean(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    weights = np.array(weights)
    values = np.array(values)
    weighted_reciprocal_sum = np.sum(weights / values)  # type: ignore
    total_weight = np.sum(weights)
    result = total_weight / weighted_reciprocal_sum
    return float(result)


def compute_ufloat_weighted_harmonic_mean(
    values: List[UFloat], weights: Optional[List[float]] = None
) -> UFloat:
    weights = np.array(weights)
    values = np.array(values)
    weighted_reciprocal_sum = np.sum(weights / values)  # type: ignore
    total_weight = np.sum(weights)
    result = total_weight / weighted_reciprocal_sum
    return result  # type: ignore


class WeightedHarmonicMeanAggregation(Aggregation):
    """
    Computes the weighted harmonic mean of a set of values with corresponding weights.

    .. math::

        H = \\frac{\\sum_{i=1}^{n} w_i}{\\sum_{i=1}^{n} \\frac{w_i}{x_i}}

    Where:
        - :math:`H` is the weighted harmonic mean
        - :math:`x_i` are the values
        - :math:`w_i` are the corresponding weights
        - :math:`n` is the number of observations


    Example:
    >>> from pumas.aggregation import aggregation_catalogue

    >>> aggregator_class = aggregation_catalogue.get("harmonic_mean")

    >>> aggregator = aggregator_class()

    >>> values = [1.0, 2.0, 3.0]
    >>> weights = [0.2, 0.3, 0.5]
    >>> result = aggregator.compute_numeric(values=values, weights=weights)
    >>> print(f"{result:.2f}")
    1.94

    >>> result = aggregator(values=values, weights=weights) # Same as compute_numeric
    >>> print(f"{result:.2f}")
    1.94

    >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
    >>> values = [ufloat(1.0, 0.1), ufloat(2.0, 0.2), ufloat(3.0, 0.3)]
    >>> weights = [0.2, 0.3, 0.5]
    >>> result = aggregator.compute_ufloat(values=values, weights=weights)
    >>> print(result)
    1.94+/-0.11
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions({})
        self._validate_and_set_parameters(params)

    def compute_numeric(
        self, values: List[float], weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute the  weighted harmonic mean for uncertain float input values.

        Args:
            values (List[float]): The list of uncertain float values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            UFloat: The computed weighted harmonic mean with uncertainty.
        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_numeric_weighted_harmonic_mean(
            values=new_values, weights=new_weights
        )

    def compute_ufloat(
        self, values: List[UFloat], weights: Optional[List[float]] = None
    ) -> UFloat:
        """
        Compute the  weighted harmonic mean for uncertain float input values.

        Args:
            values (List[UFloat]): The list of uncertain float values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            UFloat: The computed weighted harmonic mean with uncertainty.
        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        return compute_ufloat_weighted_harmonic_mean(
            values=new_values, weights=new_weights
        )

    __call__ = compute_numeric
