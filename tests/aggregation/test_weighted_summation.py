import numpy as np
import pytest

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.exceptions import (
    AggregationNegativeWeightsException,
    AggregationNullValuesWarning,
    AggregationNullWeightsWarning,
    AggregationValuesToWeightLengthMismatchException,
)
from pumas.aggregation.weighted_summation import WeightedSummationAggregation


@pytest.fixture()
def aggregation():
    return WeightedSummationAggregation()


def test_aggregation_catalogue():
    assert "summation" in aggregation_catalogue.list_items()


def test_weighted_summation_nominal_values_without_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted summation with weights set ot 1."""
    # the reference value is obtained with plain python implementation
    values, _ = dataset_1
    weights = np.ones_like(values)
    reference = sum(v * w for v, w in zip(values, weights))

    results = [
        aggregation.compute_uscore(values=values, weights=[w / w for w in weights])
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_summation_nominal_values_with_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted summation with weights."""

    values, weights = dataset_1

    # the reference value is obtained with plain python implementation
    reference = sum(v * w for v, w in zip(values, weights))

    results = [
        aggregation.compute_uscore(values, weights)
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_summation_mismatching_lengths(aggregation, dataset_5):
    values, weights = dataset_5

    with pytest.raises(AggregationValuesToWeightLengthMismatchException):
        aggregation.compute_score(values, weights)


def test_weighted_summation_negative_weights(aggregation, dataset_6):
    values, weights = dataset_6

    with pytest.raises(AggregationNegativeWeightsException):
        aggregation.compute_score(values, weights)


def test_weighted_summation_skips_null_value_or_weight_float(
    aggregation,
    dataset_17,
    dataset_8,
    dataset_9,
    dataset_12,
    dataset_13,
):
    values, weights = dataset_17
    reference_result = aggregation.compute_score(values=values, weights=weights)
    results = []
    for dataset in [
        dataset_8,
        dataset_9,
    ]:
        values, weights = dataset
        with pytest.warns(AggregationNullValuesWarning):
            result = aggregation.compute_score(values=values, weights=weights)
            results.append(result)

    for dataset in [dataset_12, dataset_13]:
        values, weights = dataset
        with pytest.warns(AggregationNullWeightsWarning):
            result = aggregation.compute_score(values=values, weights=weights)
            results.append(result)

    assert all([r == pytest.approx(reference_result) for r in results])


def test_weighted_summation_skips_null_value_or_weight_ufloat(
    aggregation, dataset_16, dataset_10, dataset_11, dataset_14, dataset_15
):
    values, weights = dataset_16
    reference_result = aggregation.compute_uscore(values=values, weights=weights)

    results = []
    for dataset in [
        dataset_10,
        dataset_11,
    ]:
        values, weights = dataset
        with pytest.warns(AggregationNullValuesWarning):
            result = aggregation.compute_uscore(values=values, weights=weights)
            results.append(result)

    for dataset in [dataset_14, dataset_15]:
        values, weights = dataset
        with pytest.warns(AggregationNullWeightsWarning):
            result = aggregation.compute_uscore(values=values, weights=weights)
            results.append(result)

    assert all(
        [
            r.nominal_value == pytest.approx(reference_result.nominal_value)
            for r in results
        ]
    )
