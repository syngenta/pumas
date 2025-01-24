from typing import Any, Dict, List

import pytest

from pumas.dataframes.dataframe import DataFrame


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    return [
        {"A": 1, "B": 2.0, "C": "x"},
        {"A": 3, "B": 4.0, "C": "y"},
        {"A": 5, "B": None, "C": "z"},
    ]


@pytest.fixture
def df(sample_data) -> DataFrame:
    return DataFrame(row_data=sample_data)


@pytest.fixture
def column_metadata_map() -> Dict[str, Dict[str, Any]]:
    return {
        "A": {"uid": "1", "properties": {"unit": "count"}},
        "B": {"uid": "2", "properties": {"unit": "count"}},
        "D": {
            "uid": "3",
            "properties": {"unit": "count"},
        },  # Column "D" not present in the data_frame
    }


def test_initialize_column_map(df):
    assert set(df.column_map.keys()) == {"A", "B", "C"}
    for column_name in df.column_map:
        assert df.column_map[column_name].name == column_name


def test_initialize_dtypes_map(df):
    assert df.dtypes_map == {
        "A": int,
        "B": float,
        "C": str,
    }


def test_initialize_column_metadata_map(df, column_metadata_map):
    # Test with provided column metadata
    metadata_map = df._initialize_column_metadata_map(column_metadata_map)
    assert set(metadata_map.keys()) == {"A", "B", "C"}
