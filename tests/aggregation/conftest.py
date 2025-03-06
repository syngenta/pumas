from typing import List, Tuple, Union

import pytest

from pumas.uncertainty.uncertainties_wrapper import UFloat, ufloat


@pytest.fixture
def values_float() -> List[float]:
    return [1.5, 2.5, 3.5]


@pytest.fixture
def values_float_short() -> List[float]:
    return [1.5, 3.5]


@pytest.fixture
def values_float_incl_negative() -> List[float]:
    return [-1.5, 2.5, 3.5]


@pytest.fixture
def values_float_incl_none() -> List[Union[float, None]]:
    return [1.5, None, 3.5]


@pytest.fixture
def values_float_incl_nan() -> List[float]:
    return [1.5, float("nan"), 3.5]


@pytest.fixture
def values_ufloat() -> List[UFloat]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=2.5, std_dev=0.2),
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def values_ufloat_incl_none() -> List[Union[UFloat, None]]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        None,
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def values_ufloat_incl_nan() -> List[Union[UFloat, float]]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        float("nan"),
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def values_ufloat_incl_nan_as_nominal_value() -> List[UFloat]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=float("nan"), std_dev=0.2),
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def values_ufloat_incl_nan_as_std_dev() -> List[UFloat]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=2.5, std_dev=float("nan")),
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def values_ufloat_short() -> List[UFloat]:
    return [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=3.5, std_dev=0.3),
    ]


@pytest.fixture
def weights_float() -> List[float]:
    return [0.5, 0.3, 0.2]


@pytest.fixture
def weights_float_short() -> List[float]:
    return [0.5, 0.2]


@pytest.fixture
def weights_float_incl_negative() -> List[float]:
    return [-0.5, 0.3, 0.2]


@pytest.fixture
def weights_float_incl_none() -> List[Union[float, None]]:
    return [0.5, None, 0.2]


@pytest.fixture
def weights_float_incl_nan() -> List[float]:
    return [0.5, float("nan"), 0.2]


@pytest.fixture
def dataset_1(
    values_float: List[float], weights_float: List[float]
) -> Tuple[List[float], List[float]]:
    return values_float, weights_float


@pytest.fixture
def dataset_2(
    values_ufloat: List[UFloat], weights_float: List[float]
) -> Tuple[List[UFloat], List[float]]:
    return values_ufloat, weights_float


# Fixture for data_frame with mismatched lengths between values and weights
@pytest.fixture
def dataset_5a(
    values_float: List[float], weights_float_short: List[float]
) -> Tuple[List[float], List[float]]:
    return values_float, weights_float_short


@pytest.fixture
def dataset_5b(
    values_float_short: List[float], weights_float: List[float]
) -> Tuple[List[float], List[float]]:
    return values_float_short, weights_float


@pytest.fixture
def dataset_5c(
    values_ufloat_short: List[UFloat], weights_float: List[float]
) -> Tuple[List[UFloat], List[float]]:
    return values_ufloat_short, weights_float


@pytest.fixture
def dataset_6(
    values_float: List[float], weights_float_incl_negative: List[float]
) -> Tuple[List[float], List[float]]:
    return values_float, weights_float_incl_negative


# Fixture for data_frame with negative values
@pytest.fixture
def dataset_7(
    values_float_incl_negative: List[Union[float, None]], weights_float: List[float]
) -> Tuple[List[Union[float, None]], List[float]]:
    return values_float_incl_negative, weights_float


@pytest.fixture
def dataset_8a(
    values_float_incl_none: List[Union[float, None]], weights_float: List[float]
) -> Tuple[List[Union[float, None]], List[float]]:
    return values_float_incl_none, weights_float


@pytest.fixture
def dataset_8b(
    values_float_incl_nan: List[Union[float, None]], weights_float: List[float]
) -> Tuple[List[Union[float, None]], List[float]]:
    return values_float_incl_nan, weights_float


@pytest.fixture
def dataset_9a(
    values_ufloat_incl_none: List[Union[UFloat, None]], weights_float: List[float]
) -> Tuple[List[Union[UFloat, None]], List[float]]:
    return values_ufloat_incl_none, weights_float


@pytest.fixture
def dataset_9b(
    values_ufloat_incl_nan: List[Union[UFloat, None]], weights_float: List[float]
) -> Tuple[List[Union[UFloat, None]], List[float]]:
    return values_ufloat_incl_nan, weights_float


@pytest.fixture
def dataset_9c(
    values_ufloat_incl_nan_as_nominal_value: List[UFloat], weights_float: List[float]
) -> Tuple[List[UFloat], List[float]]:
    return values_ufloat_incl_nan_as_nominal_value, weights_float


@pytest.fixture
def dataset_9d(
    values_ufloat_incl_nan_as_std_dev: List[UFloat], weights_float: List[float]
) -> Tuple[List[UFloat], List[float]]:
    return values_ufloat_incl_nan_as_std_dev, weights_float


@pytest.fixture
def dataset_10a(
    values_float: List[float], weights_float_incl_none: List[Union[float, None]]
) -> Tuple[List[float], List[Union[float, None]]]:
    return values_float, weights_float_incl_none


@pytest.fixture
def dataset_10b(
    values_float: List[float], weights_float_incl_nan: List[Union[float, None]]
) -> Tuple[List[float], List[Union[float, None]]]:
    return values_float, weights_float_incl_nan


@pytest.fixture
def dataset_11(
    values_float_short: List[float], weights_float_short: List[float]
) -> Tuple[List[float], List[float]]:
    return values_float_short, weights_float_short


@pytest.fixture
def dataset_12(
    values_ufloat_short: List[UFloat], weights_float_short: List[float]
) -> Tuple[List[UFloat], List[float]]:
    return values_ufloat_short, weights_float_short
