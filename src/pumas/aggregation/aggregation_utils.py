import logging
import warnings
from typing import Any, List, Tuple

from pumas.aggregation.exceptions import (
    AggregationNegativeValuesException,
    AggregationNegativeWeightsException,
    AggregationNullValuesException,
    AggregationNullValuesWarning,
    AggregationNullWeightsException,
    AggregationNullWeightsWarning,
    AggregationValuesToWeightLengthMismatchException,
)

# Configure the logging module
logger = logging.getLogger(__name__)


def is_nan_none(item: Any):
    if item is None:
        return True
    if "nan" in str(item):
        return True


def warn_and_log(message, category=Warning):
    """Send a warning message and log it.

    Args:
        message (str): The message to log and warn about.
        category (Warning, optional): The warning category.

    """
    warnings.warn(message, category)
    logging.warning(message)


def check_length_match(values: List, weights: List):
    """Check if the length of the 'values' and 'weights' arrays match.

    Args:
        values (List): The values to be aggregated.
        weights (List): The weights corresponding to the values.

    Raises:
        AggregationValuesToWeightLengthMismatchException: If the length of
        values and weights do not match.

    """
    if len(values) != len(weights):
        raise AggregationValuesToWeightLengthMismatchException(
            "The length of values and weights does not match."
        )


def check_negative_values(values: List):
    """Check if any value is negative in the 'values' array.

    Args:
        values (List): The values to be aggregated.

    Raises:
        AggregationNegativeValuesException: If any value is negative.

    """
    if any(v < 0 for v in values):
        raise AggregationNegativeValuesException("All values must be positive.")


def check_negative_weights(weights: List):
    """Check if any weight is negative in the 'weights' array.

    Args:
        weights (List): The weights corresponding to the values.

    Raises:
        AggregationNegativeWeightsException: If any weight is negative.

    """
    if any(w < 0 for w in weights):
        raise AggregationNegativeWeightsException("All values must be positive.")


def check_null_weights(weights: List):
    """Check for null or NaN weights in the 'weights' array.

    Args:
        weights (List): The weights corresponding to the values.

    Raises:
        AggregationNullWeightsException: If any weight is null or NaN.

    """
    if any(is_nan_none(w) for w in weights):
        raise AggregationNullWeightsException("All weights must be non-zero.")


def check_null_values(values: List):
    """Check for null or NaN values in the 'values' array.

    Args:
        values (List): The values to be aggregated.

    Raises:
        AggregationNullValuesException: If any value is null or NaN.

    """
    if any(is_nan_none(item=v) for v in values):
        raise AggregationNullValuesException("None or NaN values are not allowed.")


def report_null_values(values: List):
    """Report null or NaN values in the 'values' array using a warning.

    Args:
        values (List): The values to be aggregated.

    """
    if any(is_nan_none(v) for v in values):
        warn_and_log(
            category=AggregationNullValuesWarning,
            message="None or NaN values are not allowed.",
        )


def report_null_weights(weights: List):
    """Report null or NaN values in the 'weights' array using a warning.

    Args:
        weights (List): The weights corresponding to the values.

    """
    if any(is_nan_none(w) for w in weights):
        warn_and_log(
            category=AggregationNullWeightsWarning,
            message="None or NaN weights are not allowed.",
        )


def filter_out_null_values_weights_pairs(
    values: List, weights: List
) -> Tuple[List, List]:
    """Filter out (value, weight) pairs if either value or weight is  null or NaN .

    Args:
        values (List): The values to be aggregated.
        weights (List): The weights corresponding to the values.

    Returns:
        Tuple[List, List]: Arrays of non-null/non-NaN
            values and their corresponding weights.

    """

    selected_values = []
    selected_weights = []
    removed_values = []
    removed_weights = []

    for v, w in zip(values, weights):
        if is_nan_none(item=v) or is_nan_none(item=w):
            removed_values.append(v)
            removed_weights.append(w)
        else:
            selected_values.append(v)
            selected_weights.append(w)

    return selected_values, selected_weights


def run_data_validation_pipeline(values, weights):
    """Run the data_frame validation pipeline on 'values' and 'weights'.

    Data validation includes checking the length match, reporting null
    values/weights, filtering out null pairs, and checking negative values/weights.

    Args:
        values (List): The values to be aggregated.
        weights (List): The weights corresponding to the values.

    Returns:
        List, List: Validated and possibly filtered 'values' and 'weights'.

    """
    check_length_match(values=values, weights=weights)
    report_null_values(values=values)
    report_null_weights(weights=weights)
    values, weights = filter_out_null_values_weights_pairs(
        values=values, weights=weights
    )
    check_null_weights(weights=weights)
    check_null_values(values=values)
    check_negative_weights(weights=weights)
    check_negative_values(values=values)
    return values, weights
