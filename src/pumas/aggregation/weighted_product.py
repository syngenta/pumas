from typing import Iterable

import numpy as np

from pumas.aggregation.base_models import BaseAggregation


def weighted_product(values: Iterable, weights: Iterable):
    """
    Computes the weighted product of a set of values with corresponding weights.

    .. math::

        A = \\prod_{i=1}^{n} x_i^{w_i}

    Where:
        - :math:`P` is the weighted product
        - :math:`x_i` is each value in the values array
        - :math:`w_i` is the weight for each value :math:`x_i`
        - :math:`n` is the number of elements in values and weights arrays

    This model is commonly used in multi-criteria decision-making, where each criterion
    (represented by a value) is raised to the power of its weight, and then all are multiplied together.

    Args:
        values (Iterable): The values representing different criteria or attributes.
        weights (Iterable): The weights for each value, representing the importance of each criterion.

    Returns:
         Union[float,UFloat]: Resultant weighted product.

    Note:
        - Unlike the weighted geometric mean, weights are not normalized in this model.
        - The result is sensitive to the scale of the weights.
        - A weight of 0 for a criterion effectively removes it from consideration.
        - This model is useful when criteria are independent and a compensatory approach is not desired.
    """  # noqa E501

    weights = np.array(weights)
    values = np.array(values)
    weighted_sum = np.prod(values**weights)
    result = weighted_sum

    return result


class WeightedProductAggregation(BaseAggregation):
    def __init__(self):
        super().__init__(
            utility_function=weighted_product,
            coefficient_parameters_names=[],
            input_parameters_names=["values", "weights"],
        )

        attributes_change_map = {}
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
