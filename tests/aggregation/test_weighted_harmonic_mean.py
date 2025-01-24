import numpy as np
import pytest

from pumas.aggregation import aggregation_catalogue
from pumas.aggregation.exceptions import (
    AggregationNegativeValuesException,
    AggregationNegativeWeightsException,
    AggregationNullValuesWarning,
    AggregationNullWeightsWarning,
    AggregationValuesToWeightLengthMismatchException,
)
from pumas.aggregation.weighted_harmonic_mean import WeightedHarmonicMeanAggregation


@pytest.fixture()
def aggregation():
    return WeightedHarmonicMeanAggregation()


def test_aggregation_catalogue():
    assert "harmonic_mean" in aggregation_catalogue.list_items()


def test_weighted_harmonic_mean_nominal_values_without_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted harmonic mean without weights.

    the reference value is obtained with scipy.stats implementation
    from scipy.stats import hmean
    values, _ = dataset_1
    weights = np.ones_like(values)
    reference = hmean(a=values, weights=weights)
    """

    reference = 2.2183098591549295
    values, _ = dataset_1
    r = aggregation.compute_score(values, np.ones_like(values))
    results = [
        aggregation.compute_uscore(values, [w / w for w in weights])
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    assert r == pytest.approx(reference)
    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_harmonic_mean_nominal_values_with_weights(
    aggregation, dataset_1, dataset_2, dataset_3, dataset_4
):
    """Test weighted harmonic mean with weights.

    the reference value is obtained with scipy.stats implementation
    from scipy.stats import hmean
    values, weights = dataset_1
    reference = hmean(a=values, weights=weights)
    """

    reference = 1.9589552238805972
    values, weights = dataset_1
    r = aggregation.compute_score(values, weights)
    results = [
        aggregation.compute_uscore(values, weights)
        for values, weights in [dataset_2, dataset_3, dataset_4]
    ]

    nominal_values = [r.nominal_value for r in results]

    assert r == pytest.approx(reference)
    for nv in nominal_values:
        assert nv == pytest.approx(reference)


def test_weighted_harmonic_mean_mismatching_lengths(aggregation, dataset_5):
    values, weights = dataset_5

    with pytest.raises(AggregationValuesToWeightLengthMismatchException):
        aggregation.compute_score(values, weights)


def test_weighted_harmonic_mean_negative_weights(aggregation, dataset_6):
    values, weights = dataset_6

    with pytest.raises(AggregationNegativeWeightsException):
        aggregation.compute_score(values, weights)


def test_weighted_harmonic_mean_negative_values(aggregation, dataset_7):
    values, weights = dataset_7

    with pytest.raises(AggregationNegativeValuesException):
        aggregation.compute_score(values, weights)


def test_weighted_harmonic_mean_skips_null_value_or_weight_float(
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


def test_weighted_harmonic_mean_skips_null_value_or_weight_ufloat(
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
