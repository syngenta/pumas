from typing import Iterable

import numpy as np

from pumas.aggregation.base_models import BaseAggregation


def weighted_geometric_mean(values: Iterable, weights: Iterable):
    """
    Computes the weighted geometric mean of a set of values with corresponding weights.

    .. math::

        G = \\left(\\prod_{i=1}^{n} x_i^{w_i} \\right)^{\\frac{1}{\\sum_{i=1}^{n} w_i}}

    Where:
        - :math:`G` is the weighted geometric mean
        - :math:`x_i` is each value in the values array
        - :math:`w_i` is the normalized weight for each value :math:`x_i`
        - :math:`n` is the number of elements in values and weights arrays

    Args:
        values (Iterable): The values to be averaged.
        weights (Iterable): The weights for each value.

    Returns:
         Union[float,UFloat]: Resultant weighted geometric mean.
    """
    weights = np.array(weights)
    values = np.array(values)
    exponents = weights / np.sum(weights)
    result = np.prod(values**exponents)

    return result


class WeightedGeometricMeanAggregation(BaseAggregation):
    def __init__(self):
        super().__init__(
            utility_function=weighted_geometric_mean,
            coefficient_parameters_names=[],
            input_parameters_names=["values", "weights"],
        )

        attributes_change_map = {}
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
