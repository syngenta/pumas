from typing import Any, Dict, List, Optional

import numpy as np

from pumas.aggregation.aggregation_utils import run_data_validation_pipeline
from pumas.aggregation.base_models import BaseAggregation
from pumas.uncertainty.uncertainties_wrapper import UFloat


def compute_numeric_weighted_product(
    values: List[float], weights: Optional[List[float]] = None
) -> float:
    values, weights = run_data_validation_pipeline(values=values, weights=weights)
    result: float = np.prod(np.power(values, weights))
    return float(result)


def compute_ufloat_weighted_product(
    values: List[UFloat], weights: Optional[List[float]] = None
) -> UFloat:
    values, weights = run_data_validation_pipeline(values=values, weights=weights)
    result: UFloat = np.prod(np.power(values, weights))
    return result


class WeightedProductAggregation(BaseAggregation):
    """
    Computes the weighted product of a set of values with corresponding weights.

    .. math::

        A = \\prod_{i=1}^{n} x_i^{w_i}

    Where:
        - :math:`P` is the weighted product
        - :math:`x_i` is each value in the values array
        - :math:`w_i` is the weight for each value :math:`x_i`
        - :math:`n` is the number of elements in values and weights arrays

    Example:
    >>> from pumas.aggregation import aggregation_catalogue

    >>> aggregator_class = aggregation_catalogue.get("product")

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
    """  # noqa E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions({})
        self._validate_and_set_parameters(params)

    def compute_numeric(
        self, values: List[float], weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute the weighted product for numeric input values.

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
        return compute_numeric_weighted_product(values=new_values, weights=new_weights)

    def compute_ufloat(
        self, values: List[UFloat], weights: Optional[List[float]] = None
    ) -> UFloat:
        """
        Compute the weighted product for uncertain float input values.

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
        return compute_ufloat_weighted_product(values=new_values, weights=new_weights)

    __call__ = compute_numeric
