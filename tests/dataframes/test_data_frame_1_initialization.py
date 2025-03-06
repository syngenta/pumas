# type: ignore
from typing import Any, Dict, List

import pytest

from pumas.dataframes.dataframe import (
    ColumnMetadata,
    DataFrame,
    Index,
    UnspecifiedDataType,
)


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    return [
        {"A": 1, "B": 4.0, "C": "x", "D": {"p": 1, "q": 2}, "E": 1, "F": 1.0, "G": 1},
        {"A": 2, "B": 5.0, "C": "y", "D": {"p": 1, "q": 2}, "E": 2, "F": 2.0, "G": 2},
        {
            "A": 3,
            "B": 6.0,
            "C": "z",
            "D": {"p": 1, "q": 2},
            "E": 3.0,
            "F": "3",
            "G": "3",
        },
    ]


@pytest.fixture
def df(sample_data) -> DataFrame:
    return DataFrame(sample_data)


def test_empty_dataframe():
    df = DataFrame()
    assert isinstance(df, DataFrame)
    assert df.shape == (0, 0)
    assert df.columns == []
    assert df.index.values == []
    assert df.column_metadata_map == {}
    assert df.column_map == {}


def test_initialization(df):
    """
    Test the initialization of PropertyDataFrame with a pandas DataFrame.
    """
    assert isinstance(df, DataFrame)
    assert isinstance(df.index, Index)
    assert isinstance(df.column_metadata_map, dict)
    assert isinstance(df.column_map, dict)
    assert isinstance(df.dtypes_map, dict)
    assert df.column_map.keys() == df.column_metadata_map.keys() == df.dtypes_map.keys()
    assert df.columns == list(df.column_map.keys())
    assert all([isinstance(v, ColumnMetadata) for v in df.column_metadata_map.values()])
    assert df.dtypes_map == {
        "A": int,
        "B": float,
        "C": str,
        "D": dict,
        "E": UnspecifiedDataType,
        "F": UnspecifiedDataType,
        "G": UnspecifiedDataType,
    }


def test_shape(df):
    assert df.shape == (3, 7)
    assert df.num_rows == 3
    assert df.num_columns == 7
    assert df.size == 21


@pytest.fixture
def data_with_inconsistent_keys():
    return [{"A": 1, "B": 2}, {"B": 3, "C": 4}, {"A": 5, "C": 6}]


@pytest.fixture
def expected_columns():
    return {"A", "B", "C"}


@pytest.fixture
def expected_normalized_data():
    return [
        {"C": None, "A": 1, "B": 2},
        {"C": 4, "A": None, "B": 3},
        {"C": 6, "A": 5, "B": None},
    ]


@pytest.fixture
def expected_column_oriented_data():
    return {"A": [1, None, 5], "B": [2, 3, None], "C": [None, 4, 6]}


def test_initialize_column_map(data_with_inconsistent_keys, expected_columns):
    df = DataFrame(row_data=data_with_inconsistent_keys)
    assert set(df.column_map.keys()) == expected_columns


def test_normalize_data(data_with_inconsistent_keys, expected_normalized_data):
    df = DataFrame(row_data=data_with_inconsistent_keys)
    assert df.row_data == expected_normalized_data


def test_convert_to_column_oriented(
    data_with_inconsistent_keys, expected_normalized_data, expected_column_oriented_data
):
    df = DataFrame(row_data=data_with_inconsistent_keys)
    assert df.row_data == expected_normalized_data
    assert df.column_data == expected_column_oriented_data
