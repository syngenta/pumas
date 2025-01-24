import uuid
from typing import Any, Dict, List

import pytest

from pumas.dataframes.dataframe import (
    ColumnNotFoundError,
    DataFrame,
    DuplicateValuesError,
    Index,
    UnsupportedIndexCreationMethodError,
)


def test_index_creation():
    values = [1, 2, 3]
    index = Index(values)
    assert index.to_list() == values


def test_index_length():
    values = [1, 2, 3, 4, 5]
    index = Index(values)
    assert len(index) == len(values)


def test_index_getitem():
    values = ["a", "b", "c"]
    index = Index(values)
    assert index[0] == "a"
    assert index[1] == "b"
    assert index[2] == "c"


def test_index_iteration():
    values = [1, 2, 3, 4, 5]
    index = Index(values)
    assert list(index) == values


def test_index_is_unique():
    unique_values = [1, 2, 3]
    non_unique_values = [1, 1, 2]

    unique_index = Index(unique_values)
    assert unique_index.is_unique

    with pytest.raises(DuplicateValuesError):
        Index(non_unique_values)


def test_index_copy():
    values = [1, 2, 3]
    index = Index(values)
    copied_index = index.copy()

    assert copied_index.to_list() == values
    assert copied_index is not index


def test_index_from_range():
    size = 5
    index = Index.from_range(size)
    assert index.to_list() == list(range(size))


def test_index_from_uids():
    size = 3
    index = Index.from_uids(size)
    assert len(index) == size
    assert all(isinstance(val, uuid.UUID) for val in index)


def test_index_rebuild():
    size = 5
    strategy = "range"
    index = Index.rebuild(size, strategy)
    assert index.to_list() == list(range(size))

    strategy = "uuids"
    index = Index.rebuild(size, strategy)
    assert len(index) == size
    assert all(isinstance(val, uuid.UUID) for val in index)

    with pytest.raises(UnsupportedIndexCreationMethodError):
        Index.rebuild(size, "unsupported")


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    return [
        {"A": 1, "B": 4, "C": "x", "D": 1},
        {"A": 2, "B": 5, "C": "y", "D": 2},
        {"A": 3, "B": 6, "C": "y", "D": None},
    ]


@pytest.fixture
def df(sample_data) -> DataFrame:
    return DataFrame(sample_data)


def test_initialization_with_unique_index(df):
    assert df.index.is_unique


def test_set_index_from_existing_column(df):
    df.set_index_from_column(column_name="A")
    assert df.index.to_list() == [1, 2, 3]


def test_set_index_from_nonexistent_column(df):
    with pytest.raises(ColumnNotFoundError):
        df.set_index_from_column(column_name="F")


def test_set_index_from_existing_column_with_duplicates(df):
    with pytest.raises(DuplicateValuesError):
        df.set_index_from_column("C")


def test_rebuild_index(df):
    df.rebuild_index(strategy="range")
    assert df.index.to_list() == [0, 1, 2]


def test_check_column_has_unique_value(df):
    assert df.column_has_unique_values("A")
    assert not df.column_has_unique_values("C")


def test_set_index_from_existing_column_with_none(df):
    with pytest.raises(ValueError):
        df.set_index_from_column("D")
