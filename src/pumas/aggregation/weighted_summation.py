from typing import Iterable

import numpy as np

from pumas.aggregation.base_models import BaseAggregation


def weighted_summation(values: Iterable, weights: Iterable):
    """
    Computes the weighted summation of a set of values with corresponding weights.

    .. math::

        A = \\sum_{i=1}^{n}{w_i x_i}

    Where:
        - :math:`A` is the weighted summation
        - :math:`x_i` is each value in the values array
        - :math:`w_i` is the weight corresponding to each value :math:`x_i`
        - :math:`n` is the number of elements in the values and weights arrays

    Args:
        values (Iterable): The values to be averaged.
        weights (Iterable): The weights for each value.

    Returns:
        Union[float,UFloat]: The weighted arithmetic mean.
    """
    values = np.array(values)
    weights = np.array(weights)
    weighted_sum = np.sum(values * weights)
    result = weighted_sum
    return result


class WeightedSummationAggregation(BaseAggregation):
    def __init__(self):
        super().__init__(
            utility_function=weighted_summation,
            coefficient_parameters_names=[],
            input_parameters_names=["values", "weights"],
        )

        attributes_change_map = {}
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
