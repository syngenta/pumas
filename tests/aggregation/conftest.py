import pytest
from uncertainties import ufloat


# Fixture for values without uncertainties and weights without uncertainties
@pytest.fixture
def dataset_1():
    values = [1.5, 2.5, 3.5]
    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture values with uncertainties and weights without
@pytest.fixture
def dataset_2():
    values = [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=2.5, std_dev=0.2),
        ufloat(nominal_value=3.5, std_dev=0.1),
    ]
    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for values without uncertainties and weights with uncertainties
@pytest.fixture
def dataset_3():
    values = [1.5, 2.5, 3.5]

    weights = [
        ufloat(nominal_value=0.5, std_dev=0.01),
        ufloat(nominal_value=0.3, std_dev=0.01),
        ufloat(nominal_value=0.2, std_dev=0.005),
    ]
    return values, weights


# Fixture for values with uncertainties and weights with uncertainties
@pytest.fixture
def dataset_4():
    values = [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=2.5, std_dev=0.2),
        ufloat(nominal_value=3.5, std_dev=0.1),
    ]

    weights = [
        ufloat(nominal_value=0.5, std_dev=0.01),
        ufloat(nominal_value=0.3, std_dev=0.01),
        ufloat(nominal_value=0.2, std_dev=0.005),
    ]
    return values, weights


# Fixture for data_frame with mismatched lengths between values and weights
@pytest.fixture
def dataset_5():
    values = [1.5, 2.5, 3.5]
    weights = [0.5, 0.3]
    return values, weights


# Fixture for data_frame with negative weights
@pytest.fixture
def dataset_6():
    values = [1.5, 2.5, 3.5]
    weights = [-0.5, 0.3, 0.2]
    return values, weights


# Fixture for data_frame with negative values
@pytest.fixture
def dataset_7():
    values = [-1.5, 2.5, 3.5]
    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for data_frame with None values
@pytest.fixture
def dataset_8():
    values = [1.5, None, 3.5]
    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for data_frame with float(nan) values
@pytest.fixture
def dataset_9():
    values = [1.5, float("nan"), 3.5]
    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for data_frame with ufloat(nan, x) values
@pytest.fixture
def dataset_10():
    nan = float("nan")

    values = [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=nan, std_dev=0.2),
        ufloat(nominal_value=3.5, std_dev=0.1),
    ]

    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for data_frame with ufloat(x, nan) values
@pytest.fixture
def dataset_11():
    nan = float("nan")
    values = [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=2.5, std_dev=nan),
        ufloat(nominal_value=3.5, std_dev=0.1),
    ]

    weights = [0.5, 0.3, 0.2]
    return values, weights


# Fixture for values without uncertainties and weights without uncertainties
@pytest.fixture
def dataset_12():
    values = [1.5, 2.5, 3.5]
    weights = [0.5, None, 0.2]
    return values, weights


# Fixture for values without uncertainties and weights without uncertainties
@pytest.fixture
def dataset_13():
    nan = float("nan")
    values = [1.5, 2.5, 3.5]
    weights = [0.5, nan, 0.2]
    return values, weights


# Fixture for data_frame with ufloat(nan, x) weights
@pytest.fixture
def dataset_14():
    nan = float("nan")
    values = [1.5, 2.5, 3.5]
    weights = [
        ufloat(nominal_value=0.5, std_dev=0.01),
        ufloat(nominal_value=nan, std_dev=0.01),
        ufloat(nominal_value=0.2, std_dev=0.005),
    ]
    return values, weights


# Fixture for data_frame with ufloat(nx, nan) weights
@pytest.fixture
def dataset_15():
    nan = float("nan")
    values = [1.5, 2.5, 3.5]

    weights = [
        ufloat(nominal_value=0.5, std_dev=0.01),
        ufloat(nominal_value=0.3, std_dev=nan),
        ufloat(nominal_value=0.2, std_dev=0.005),
    ]
    return values, weights


# Fixture for values and weights with uncertainty
# and with fewer data (removing the nan/non-nan pair of other fixtures)
@pytest.fixture
def dataset_16():
    values = [
        ufloat(nominal_value=1.5, std_dev=0.1),
        ufloat(nominal_value=3.5, std_dev=0.1),
    ]

    weights = [
        ufloat(nominal_value=0.5, std_dev=0.01),
        ufloat(nominal_value=0.2, std_dev=0.005),
    ]
    return values, weights


# Fixture for values and weights with uncertainty
# and with fewer data (removing the nan/non-nan pair of other fixtures)
@pytest.fixture
def dataset_17():
    values = [1.5, 3.5]
    weights = [0.5, 0.2]
    return values, weights
