from typing import Iterable

import numpy as np

from pumas.aggregation.base_models import BaseAggregation


def weighted_harmonic_mean(values: Iterable, weights: Iterable):
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


class WeightedHarmonicMeanAggregation(BaseAggregation):
    def __init__(self):
        super().__init__(
            utility_function=weighted_harmonic_mean,
            coefficient_parameters_names=[],
            input_parameters_names=["values", "weights"],
        )

        attributes_change_map = {}
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
