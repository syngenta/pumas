from typing import Any, Dict, List

import pytest

from pumas.dataframes.dataframe import DataFrame
from pumas.dataframes.exceptions import ColumnNotFoundError


def add(x, amount):
    if x is not None:
        return x + amount
    return None


def square(x):
    if x is not None:
        return x**2
    return None


@pytest.fixture
def data() -> List[Dict[str, Any]]:
    return [
        {"A": 1, "B": 2, "C": 3},
        {"A": 4, "B": 5, "C": 6},
        {"A": 7, "B": 8, "C": 9},
    ]


@pytest.fixture
def scrambled_data() -> List[Dict[str, Any]]:
    return [
        {"A": 1, "B": 2, "C": 3},
        {"A": 7, "B": 8, "C": 9},
        {"A": 4, "B": 5, "C": 6},
    ]


def test_apply_elementwise_column_serial(data):
    df = DataFrame(row_data=data)
    func_kwargs = {"amount": 2}
    result_df = df.apply_elementwise_column(
        column_name="A",
        new_column_name="A_added",
        func=add,
        num_jobs=0,
        method="threads",
        func_kwargs=func_kwargs,
    )
    expected_data = [{"A_added": 3}, {"A_added": 6}, {"A_added": 9}]
    assert result_df.row_data == expected_data
    assert result_df.index.values == df.index.values
    assert result_df.columns == ["A_added"]
    assert result_df.shape == (3, 1)


def test_apply_elementwise_column_threads(data):
    df = DataFrame(row_data=data)
    func_kwargs = {"amount": 2}
    result_df = df.apply_elementwise_column(
        column_name="A",
        new_column_name="A_added",
        func=add,
        num_jobs=2,
        method="threads",
        func_kwargs=func_kwargs,
    )
    expected_data = [{"A_added": 3}, {"A_added": 6}, {"A_added": 9}]
    assert result_df.row_data == expected_data
    assert result_df.index.values == df.index.values
    assert result_df.columns == ["A_added"]
    assert result_df.shape == (3, 1)


def test_apply_elementwise_column_processes(data):
    df = DataFrame(row_data=data)
    func_kwargs = {"amount": 2}
    result_df = df.apply_elementwise_column(
        column_name="A",
        new_column_name="A_added",
        func=add,
        num_jobs=2,
        method="processes",
        func_kwargs=func_kwargs,
    )
    expected_data = [{"A_added": 3}, {"A_added": 6}, {"A_added": 9}]
    assert result_df.row_data == expected_data
    assert result_df.index.values == df.index.values
    assert result_df.columns == ["A_added"]
    assert result_df.shape == (3, 1)


def test_apply_elementwise_column_invalid_method(data):
    df = DataFrame(row_data=data)
    func_kwargs = {"amount": 2}
    with pytest.raises(ValueError):
        df.apply_elementwise_column(
            column_name="A",
            new_column_name="A_added",
            func=add,
            num_jobs=2,
            method="invalid",
            func_kwargs=func_kwargs,
        )


def test_apply_elementwise_column_nonexistent_column(data):
    df = DataFrame(row_data=data)
    func_kwargs = {"amount": 2}
    with pytest.raises(ColumnNotFoundError):
        df.apply_elementwise_column(
            column_name="D",
            new_column_name="D_added",
            func=add,
            num_jobs=2,
            method="threads",
            func_kwargs=func_kwargs,
        )


def test_apply_elementwise_column_scrambled_index(scrambled_data):
    df = DataFrame(row_data=scrambled_data)
    df.rebuild_index(strategy="uuids")  # Rebuild index to have non-sequential order
    func_kwargs = {"amount": 2}
    result_df = df.apply_elementwise_column(
        column_name="A",
        new_column_name="A_added",
        func=add,
        num_jobs=2,
        method="threads",
        func_kwargs=func_kwargs,
    )
    expected_data = [{"A_added": 3}, {"A_added": 9}, {"A_added": 6}]
    assert result_df.row_data == expected_data
    assert result_df.index.values == df.index.values
    assert result_df.columns == ["A_added"]
    assert result_df.shape == (3, 1)


def test_apply_elementwise_column_square(data):
    df = DataFrame(row_data=data)
    result_df = df.apply_elementwise_column(
        column_name="A",
        new_column_name="A_squared",
        func=square,
        num_jobs=2,
        method="threads",
    )
    expected_data = [{"A_squared": 1}, {"A_squared": 16}, {"A_squared": 49}]
    assert result_df.row_data == expected_data
    assert result_df.index.values == df.index.values
    assert result_df.columns == ["A_squared"]
    assert result_df.shape == (3, 1)
