from typing import Any, Dict, List, Optional

import numpy as np

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.aggregation.base_models import BaseAggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat


def compute_numeric_weighted_deviation_index(
    values: List[float], weights: Optional[List[float]] = None, ideal_value: float = 1.0
) -> float:
    weights = np.array(weights)
    values = np.array(values)
    weight_squared_sum = np.sum(weights**2)  # type: ignore
    deltas = ideal_value - values  # type: ignore

    sum_term = np.sum(weights**2 * deltas**2)  # type: ignore
    result = 1.0 - np.sqrt(sum_term / weight_squared_sum)

    return float(result)


def compute_ufloat_weighted_deviation_index(
    values: List[UFloat],
    weights: Optional[List[float]] = None,
    ideal_value: float = 1.0,
) -> UFloat:
    weights = np.array(weights)
    values = np.array(values)
    weight_squared_sum = np.sum(weights**2)  # type: ignore
    deltas = ideal_value - values  # type: ignore
    sum_term = np.sum(weights**2 * deltas**2)  # type: ignore
    result: UFloat = 1.0 - (sum_term / weight_squared_sum) ** 0.5

    return result


class WeightedDeviationIndexAggregation(BaseAggregation):
    """
     Computes the weighted deviation index of a set of values with corresponding weights.

     This approach to aggregations combines multiple values while accounting for their deviation from an ideal reference value.
     This method evaluates the overall score by penalizing the squared deviation of each value from the specified
     ideal value, amplifying the importance of each deviation by the associated weight. It is designed to be
     insensitive to the amount of data, i.e., it works for varying numbers of criteria and is robust to missing data
     by adjusting the dimensionality of the analysis based on the available data_frame.


     .. math::

         D = 1 - \\sqrt{\\frac{\\sum{w_i^2 (x_{\\text{ideal}} - x_i)^2}}{\\sum{w_i^2}}}

     Where:
         - :math:`D` is the weighted deviation index
         - :math:`x_i` are the values
         - :math:`w_i` are the corresponding weights for each value
         - :math:`{x_\\text{ideal}}` is the ideal value
         - Both :math:`x_i` and :math:`w_i` must be non-negative, and :math:`x_i` must not exceed :math:`ideal`.

     Parameters:
         params (Optional[Dict[str, Any]]): Initial parameters for the double sigmoid function. Defaults to None.

     Attributes:
         x_ideal (float): is the ideal value. Deafults to 1.0.

    Usage Example:

     >>> from pumas.aggregation import aggregation_catalogue

     >>> aggregator_class = aggregation_catalogue.get("deviation_index")

     >>> aggregator = aggregator_class()

     >>> values = [1.0, 2.0, 3.0]
     >>> weights = [0.2, 0.3, 0.5]
     >>> result = aggregator.compute_numeric(values=values, weights=weights)
     >>> print(f"{result:.2f}")
     -0.69

     >>> result = aggregator(values=values, weights=weights) # Same as compute_numeric
     >>> print(f"{result:.2f}")
     -0.69

     >>> from pumas.uncertainty.uncertainties_wrapper import ufloat
     >>> values = [ufloat(1.0, 0.1), ufloat(2.0, 0.2), ufloat(3.0, 0.3)]
     >>> weights = [0.2, 0.3, 0.5]
     >>> result = aggregator.compute_ufloat(values=values, weights=weights)
     >>> print(result)
     -0.69+/-0.23
    """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "ideal_value": {
                    "type": "float",
                    "min": float("-inf"),
                    "max": float("inf"),
                    "default": 1.0,
                },
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(
        self, values: List[float], weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute the weighted deviation index for numeric input values.

        Args:
            values (List[float]): The list of numeric values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            float: The computed weighted deviation index.

        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        parameters = self.get_parameters_values()
        return compute_numeric_weighted_deviation_index(
            values=new_values, weights=new_weights, **parameters
        )

    def compute_ufloat(
        self, values: List[UFloat], weights: Optional[List[float]] = None
    ) -> UFloat:
        """
        Compute the weighted deviation index for uncertain float input values.

        Args:
            values (List[UFloat]): The list of uncertain float values to be aggregated.
            weights (Optional[List[float]]): The list of weights corresponding to each value.
                If None, equal weights are assumed.

        Returns:
            UFloat: The computed weighted deviation index with uncertainty.
        """  # noqa: E501
        new_values, new_weights = run_data_validation_pipeline(
            values=values, weights=weights
        )
        parameters = self.get_parameters_values()
        return compute_ufloat_weighted_deviation_index(
            values=new_values, weights=new_weights, **parameters
        )

    __call__ = compute_numeric
