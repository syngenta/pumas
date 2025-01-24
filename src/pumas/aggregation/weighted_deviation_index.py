from typing import Iterable

import numpy as np

from pumas.aggregation.base_models import BaseAggregation


def weighted_deviation_index(values: Iterable, weights: Iterable, ideal_value: float):
    """
    Computes the weighted deviation index by aggregating multiple values while accounting for their
    deviation from an ideal reference value.

    This method evaluates the overall score by penalizing the squared deviation of each value from the specified
    ideal value, amplifying the importance of each deviation by the associated weight. It is designed to be
    insensitive to the amount of data_frame, i.e., it works for varying numbers of criteria and is robust to missing data_frame
    by adjusting the dimensionality of the analysis based on the available data_frame. The final score is normalized
    to range between 0 and 1, providing a consistent scale regardless of the number of values or magnitude of
    weights.

        The equation for this aggregation method is as follows:

        .. math::

            D = 1 - \\sqrt{\\frac{\\sum{w_i^2 (x_{\\text{ideal}} - x_i)^2}}{\\sum{w_i^2}}}

        Where:
            - :math:`D` is the weighted deviation index
            - :math:`x_i` are the values
            - :math:`w_i` are the corresponding weights for each value
            - :math:`{x_\\text{ideal}}` is the ideal value
            - Both :math:`x_i` and :math:`w_i` must be non-negative, and :math:`x_i` must not exceed :math:`ideal`.

        Args:
            values (Iterable): The values to be averaged.
            weights (Iterable): The weights for each value.
            ideal_value (float): The ideal or maximum desirability score possible for each criterion.

        Returns:
            Union[float,UFloat]: Resultant weighted deviation index.
    """  # noqa: E501

    weights = np.array(weights)
    values = np.array(values)
    weight_squared_sum = np.sum(weights**2)
    deltas = np.abs(values - ideal_value)

    nissink_sum_term = np.sum(weights**2 * deltas**2)
    result = 1.0 - (nissink_sum_term / weight_squared_sum) ** 0.5

    return result


class WeightedDeviationIndexAggregation(BaseAggregation):
    def __init__(self):
        super().__init__(
            utility_function=weighted_deviation_index,
            coefficient_parameters_names=["ideal_value"],
            input_parameters_names=["values", "weights"],
        )

        attributes_change_map = {
            "ideal_value": {"min": 0.0, "max": 1.0, "default": 1.0}
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
