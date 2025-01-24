import warnings

import pytest

from pumas.dataframes.dataframe import DataFrame, UnspecifiedDataType


@pytest.fixture
def data():
    return [
        {"A": "1", "B": "2.0", "C": "x", "D": 1},
        {"A": "2", "B": "3.5", "C": "y", "D": "1"},
        {"A": "3", "B": "4.2", "C": "z", "D": "x"},
    ]


@pytest.fixture
def expected_column_oriented_data_initial():
    return {
        "A": ["1", "2", "3"],
        "B": ["2.0", "3.5", "4.2"],
        "C": ["x", "y", "z"],
        "D": [1, "1", "x"],
    }


@pytest.fixture
def dtype_map_total_casting():
    return {"A": int, "B": float, "C": str, "D": str}


@pytest.fixture
def expected_column_oriented_data_total_casting():
    return {
        "A": [1, 2, 3],
        "B": [2.0, 3.5, 4.2],
        "C": ["x", "y", "z"],
        "D": ["1", "1", "x"],
    }


@pytest.fixture
def dtype_map_partial_casting():
    return {"A": int}


@pytest.fixture
def expected_column_oriented_data_partial_casting():
    return {
        "A": [1, 2, 3],
        "B": ["2.0", "3.5", "4.2"],
        "C": ["x", "y", "z"],
        "D": [1, "1", "x"],
    }


@pytest.fixture
def dtype_map_failing_casting():
    return {"C": float}


def test_initial_data(data, expected_column_oriented_data_initial):
    df = DataFrame(row_data=data)
    assert df.column_data == expected_column_oriented_data_initial

    assert df.dtypes_map["A"] == str
    assert df.dtypes_map["B"] == str
    assert df.dtypes_map["C"] == str
    assert df.dtypes_map["D"] == UnspecifiedDataType


def test_total_casting(
    data, dtype_map_total_casting, expected_column_oriented_data_total_casting
):
    df = DataFrame(row_data=data, dtypes_map=dtype_map_total_casting)
    assert df.column_data == expected_column_oriented_data_total_casting
    assert df.dtypes_map["A"] == int
    assert df.dtypes_map["B"] == float
    assert df.dtypes_map["C"] == str
    assert df.dtypes_map["D"] == str


def test_partial_casting(
    data, dtype_map_partial_casting, expected_column_oriented_data_partial_casting
):
    df = DataFrame(row_data=data, dtypes_map=dtype_map_partial_casting)
    assert df.column_data == expected_column_oriented_data_partial_casting
    assert df.dtypes_map["A"] == int
    assert df.dtypes_map["B"] == str
    assert df.dtypes_map["C"] == str
    assert df.dtypes_map["D"] == UnspecifiedDataType


def test_failing_casting(
    data, dtype_map_failing_casting, expected_column_oriented_data_initial
):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = DataFrame(row_data=data, dtypes_map=dtype_map_failing_casting)
        assert df.column_data == expected_column_oriented_data_initial
        assert df.dtypes_map["A"] == str
        assert len(w) > 0
        assert "Failed to cast column" in str(w[-1].message)


def test_column_not_found(data):
    dtype_map = {"Q": int}
    df = DataFrame(row_data=data)
    df._apply_dtype_map(dtype_map)
    # Ensure that no changes were made and no errors raised
    assert df.dtypes_map["A"] == str
    assert df.dtypes_map["B"] == str
    assert df.dtypes_map["C"] == str
