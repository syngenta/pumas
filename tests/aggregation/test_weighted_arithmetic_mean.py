import numpy as np
import pytest

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.exceptions import (
    AggregationNegativeWeightsException,
    AggregationNullValuesWarning,
    AggregationNullWeightsWarning,
    AggregationValuesToWeightLengthMismatchException,
)
from pumas.aggregation.weighted_arithmetic_mean import (
    WeightedArithmeticMeanAggregation,
    weighted_arithmetic_mean,
)


@pytest.fixture()
def aggregation():
    return WeightedArithmeticMeanAggregation()


def test_aggregation_catalogue():
    assert "arithmetic_mean" in aggregation_catalogue.list_items()


def test_weighted_arithmetic_mean_nominal_values_without_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted arithmetic mean with weights set ot 1."""
    # the reference value is obtained with numpy implementation
    values, _ = dataset_1
    reference = np.average(values, weights=np.ones_like(values))

    results = [
        aggregation.compute_uscore(values, [w / w for w in weights])
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_arithmetic_mean_nominal_values_with_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted arithmetic mean with weights.

    # the reference value is obtained with numpy implementation
    """

    values, weights = dataset_1
    reference = np.average(values, weights=weights)

    results = [
        aggregation.compute_uscore(values, weights)
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_arithmetic_mean_mismatching_lengths(aggregation, dataset_5):
    values, weights = dataset_5

    with pytest.raises(AggregationValuesToWeightLengthMismatchException):
        aggregation.compute_score(values, weights)


def test_weighted_arithmetic_mean_negative_weights(aggregation, dataset_6):
    values, weights = dataset_6

    with pytest.raises(AggregationNegativeWeightsException):
        aggregation.compute_score(values, weights)


def test_weighted_arithmetic_mean_compare_reinvent(aggregation):
    data1 = np.array([-4.43, -1.94, -2.93, 8.50, 3.71], dtype=np.float32)
    weight1 = 2.71828

    data2 = np.array([0.90, -7.40, 0.47, 2.29, -1.94], dtype=np.float32)
    weight2 = 3.14159
    data = [(data1, weight1), (data2, weight2)]

    scores = np.array([score for score, _ in data])
    weights = np.array([weight for _, weight in data])
    weights = np.array(
        np.broadcast_to(weights.reshape(-1, 1), scores.shape), dtype=np.float32
    )

    scores = scores.T
    weights = weights.T

    results = np.array(
        [weighted_arithmetic_mean(values=s, weights=w) for s, w in zip(scores, weights)]
    )
    assert np.allclose(results, [-1.57, -4.86, -1.10, 5.17, 0.68], atol=1e-2)


def test_weighted_arithmetic_mean_skips_null_value_or_weight_float(
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


def test_weighted_arithmetic_mean_skips_null_value_or_weight_ufloat(
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
