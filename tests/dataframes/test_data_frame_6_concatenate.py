import pytest

from pumas.dataframes.dataframe import DataFrame
from pumas.dataframes.dataframe_utils import concat


@pytest.fixture
def df1():
    return DataFrame(
        row_data=[
            {"A": "A0", "B": "B0", "C": "C0", "D": "D0"},
            {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
            {"A": "A2", "B": "B2", "C": "C2", "D": "D2"},
            {"A": "A3", "B": "B3", "C": "C3", "D": "D3"},
        ],
        index=[0, 1, 2, 3],
    )


@pytest.fixture
def df2():
    return DataFrame(
        row_data=[
            {"A": "A4", "B": "B4", "C": "C4", "D": "D4"},
            {"A": "A5", "B": "B5", "C": "C5", "D": "D5"},
            {"A": "A6", "B": "B6", "C": "C6", "D": "D6"},
            {"A": "A7", "B": "B7", "C": "C7", "D": "D7"},
        ],
        index=[4, 5, 6, 7],
    )


@pytest.fixture
def df3():
    return DataFrame(
        row_data=[
            {"A": "A8", "B": "B8", "C": "C8", "D": "D8"},
            {"A": "A9", "B": "B9", "C": "C9", "D": "D9"},
            {"A": "A10", "B": "B10", "C": "C10", "D": "D10"},
            {"A": "A11", "B": "B11", "C": "C11", "D": "D11"},
        ],
        index=[8, 9, 10, 11],
    )


@pytest.fixture
def df4():
    return DataFrame(
        row_data=[
            {
                "B": "B2",
                "D": "D2",
                "F": "F2",
            },
            {
                "B": "B3",
                "D": "D3",
                "F": "F3",
            },
            {
                "B": "B6",
                "D": "D6",
                "F": "F6",
            },
            {
                "B": "B7",
                "D": "D7",
                "F": "F7",
            },
        ],
        index=[2, 3, 6, 7],
    )


@pytest.fixture
def df5():
    return DataFrame(
        row_data=[
            {"F": "F0", "G": "G0", "H": "H0", "I": "I0"},
            {"F": "F1", "G": "G1", "H": "H1", "I": "I1"},
            {"F": "F2", "G": "G2", "H": "H2", "I": "I2"},
            {"F": "F3", "G": "G3", "H": "H3", "I": "I3"},
        ],
        index=[0, 1, 2, 3],
    )


def test_concat_rows_outer(df1, df2, df3):
    df_concat = concat(dataframes=[df1, df2, df3], join="outer", axis=0)
    expected_data = [
        {"A": "A0", "B": "B0", "C": "C0", "D": "D0"},
        {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
        {"A": "A2", "B": "B2", "C": "C2", "D": "D2"},
        {"A": "A3", "B": "B3", "C": "C3", "D": "D3"},
        {"A": "A4", "B": "B4", "C": "C4", "D": "D4"},
        {"A": "A5", "B": "B5", "C": "C5", "D": "D5"},
        {"A": "A6", "B": "B6", "C": "C6", "D": "D6"},
        {"A": "A7", "B": "B7", "C": "C7", "D": "D7"},
        {"A": "A8", "B": "B8", "C": "C8", "D": "D8"},
        {"A": "A9", "B": "B9", "C": "C9", "D": "D9"},
        {"A": "A10", "B": "B10", "C": "C10", "D": "D10"},
        {"A": "A11", "B": "B11", "C": "C11", "D": "D11"},
    ]
    assert df_concat.row_data == expected_data
    assert df_concat.shape == (12, 4)


def test_concat_columns_outer(df1, df5):
    df_concat = concat(dataframes=[df1, df5], join="outer", axis=1)
    expected_data = [
        {
            "A": "A0",
            "B": "B0",
            "C": "C0",
            "D": "D0",
            "F": "F0",
            "G": "G0",
            "H": "H0",
            "I": "I0",
        },
        {
            "A": "A1",
            "B": "B1",
            "C": "C1",
            "D": "D1",
            "F": "F1",
            "G": "G1",
            "H": "H1",
            "I": "I1",
        },
        {
            "A": "A2",
            "B": "B2",
            "C": "C2",
            "D": "D2",
            "F": "F2",
            "G": "G2",
            "H": "H2",
            "I": "I2",
        },
        {
            "A": "A3",
            "B": "B3",
            "C": "C3",
            "D": "D3",
            "F": "F3",
            "G": "G3",
            "H": "H3",
            "I": "I3",
        },
    ]
    assert df_concat.row_data == expected_data
    assert df_concat.shape == (4, 8)
