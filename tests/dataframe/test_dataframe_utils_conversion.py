import pytest

from pumas.dataframes.dataframe_utils import convert_columnar_to_row_data


@pytest.fixture
def columnar_data():
    return {
        "uid": ["A", "B", "C", "D", "E"],
        "quality": [1.0, 5.0, 3.0, 6.0, 2.0],
        "cost": [30.0, 40.0, 10.0, 60.0, 80.0],
        "efficiency": [0.7, 0.2, 0.9, 0.4, 0.5],
    }


def test_convert_columnar_to_row_data(columnar_data):
    expected_row_data = [
        {"uid": "A", "quality": 1.0, "cost": 30.0, "efficiency": 0.7},
        {"uid": "B", "quality": 5.0, "cost": 40.0, "efficiency": 0.2},
        {"uid": "C", "quality": 3.0, "cost": 10.0, "efficiency": 0.9},
        {"uid": "D", "quality": 6.0, "cost": 60.0, "efficiency": 0.4},
        {"uid": "E", "quality": 2.0, "cost": 80.0, "efficiency": 0.5},
    ]
    row_data = convert_columnar_to_row_data(columnar_data)
    assert row_data == expected_row_data
