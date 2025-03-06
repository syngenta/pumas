from math import isclose

import pytest

from pumas.desirability.multistep import CoordinateManager, Point


def test_point_init():
    """
    Test that the Point object is initialized correctly with given x and y coordinates.
    """  # noqa E501
    p = Point(x=1.0, y=2.0)
    assert p.x == 1.0
    assert p.y == 2.0


def test_point_repr():
    """
    Test the string representation of the Point object.
    """  # noqa E501
    p = Point(x=1.0, y=2.0)
    assert repr(p) == "Point(x=1.0, y=2.0)"


def test_point_eq_same_point():
    """
    Test equality for two Point objects with the same coordinates.
    """  # noqa E501
    p1 = Point(x=1.0, y=2.0)
    p2 = Point(x=1.0, y=2.0)
    assert p1 == p2


def test_point_eq_different_point():
    """
    Test inequality for two Point objects with different coordinates.
    """  # noqa E501
    p1 = Point(x=1.0, y=2.0)
    p2 = Point(x=2.0, y=3.0)
    assert p1 != p2


def test_point_eq_different_types():
    """
    Test equality comparison between a Point object and a non-Point object.
    """  # noqa E501
    p = Point(x=1.0, y=2.0)
    assert p != (1.0, 2.0)


def test_point_eq_float_precision():
    """
    Test equality of Point objects with very close floating-point values.
    This test ensures that small floating-point differences are handled correctly.
    """  # noqa E501
    p1 = Point(x=1.0, y=0.3333333333333333)
    p2 = Point(x=1.0, y=0.3333333333333334)
    assert p1 == p2
    assert isclose(p1.y, p2.y)


def test_point_hash():
    """
    Test that two Point objects with the same coordinates have the same hash.
    """  # noqa E501
    p1 = Point(x=1.0, y=2.0)
    p2 = Point(x=1.0, y=2.0)
    assert hash(p1) == hash(p2)


def test_point_invalid_coordinates():
    """
    Test that Point initialization raises a ValueError for non-finite coordinates.
    """  # noqa E501
    with pytest.raises(ValueError):
        Point(x=float("inf"), y=2.0)
    with pytest.raises(ValueError):
        Point(x=1.0, y=float("nan"))


def test_coordinate_manager_empty_list():
    """
    Test that CoordinateManager raises a ValueError when initialized with an empty list.
    """  # noqa E501
    with pytest.raises(ValueError, match="Coordinates list cannot be empty."):
        CoordinateManager([])


def test_coordinate_manager_single_coordinate():
    """
    Test that CoordinateManager raises a ValueError when initialized with a single coordinate.
    At least two coordinates are required to define a valid multistep function.
    """  # noqa E501
    with pytest.raises(
        ValueError,
        match="At least two coordinates are required to form a valid multistep.",
    ):
        CoordinateManager([(1.0, 2.0)])


def test_coordinate_manager_single_coordinate_after_deduplication():
    """
    Test that CoordinateManager raises a ValueError when initialized with a single coordinate.
    At least two coordinates are required to define a valid multistep function.
    """  # noqa E501
    with pytest.raises(
        ValueError,
        match="At least two DIFFERENT coordinates "
        "are required to form a valid multistep.",
    ):
        CoordinateManager([(1.0, 2.0), (1.0, 2.0)])


def test_coordinate_manager_invalid_coordinates():
    """
    Test that CoordinateManager raises a ValueError when given invalid coordinate values.
    This includes non-numeric values and mixed valid/invalid coordinates.
    """  # noqa E501
    with pytest.raises(ValueError, match="Error converting coordinates to Point: "):
        CoordinateManager([(1.0, 2.0), (1.0, "invalid"), (3.0, 4.0)])  # type: ignore
    with pytest.raises(ValueError, match="Error converting coordinates to Point: "):
        CoordinateManager([(1.0, 2.0), ("invalid", 0.1), (3.0, 4.0)])  # type: ignore
    with pytest.raises(ValueError, match="Error converting coordinates to Point: "):
        CoordinateManager([(1.0, 2.0), ("invalid", "invalid"), (3.0, 4.0)])  # type: ignore  # noqa E501


def test_coordinate_manager_duplicate_x_coordinates():
    """
    Test that CoordinateManager raises a ValueError when given duplicate x-coordinates.
    Duplicate x-coordinates would lead to ambiguity in the multistep function.
    """  # noqa E501
    with pytest.raises(ValueError):
        CoordinateManager([(1.0, 2.0), (1.0, 3.0), (3.0, 4.0)])


def test_coordinate_manager_out_of_bounds_y_coordinates():
    """
    Test that CoordinateManager raises a ValueError when y-coordinates are out of the [0, 1] range.
    Y-coordinates must be between 0 and 1 for a valid desirability function.
    """  # noqa E501
    with pytest.raises(ValueError):
        CoordinateManager([(1.0, 2.0), (3.0, 4.0), (5.0, 1.1)])
    with pytest.raises(ValueError):
        CoordinateManager([(1.0, 2.0), (3.0, 4.0), (5.0, -0.1)])


def test_coordinate_manager_valid_coordinates():
    """
    Test that CoordinateManager correctly initializes with valid coordinates.
    """  # noqa E501
    coordinates = [(1.0, 1.0), (3.0, 0.4), (5.0, 0.0)]
    manager = CoordinateManager(coordinates)
    assert len(manager.points) == 3

    expected_points = {Point(x=1.0, y=1.0), Point(x=3.0, y=0.4), Point(x=5.0, y=0.0)}
    assert set(manager.points) == expected_points


def test_coordinate_manager_sorting():
    """
    Test that CoordinateManager correctly sorts the points by x-coordinate.
    """  # noqa E501
    coordinates = [(5.0, 0.0), (1.0, 1.0), (3.0, 0.4)]
    manager = CoordinateManager(coordinates)
    sorted_points = list(manager.points)
    assert [p.x for p in sorted_points] == [1.0, 3.0, 5.0]


def test_coordinate_manager_float_precision():
    """
    Test that CoordinateManager handles floating-point precision correctly.
    """  # noqa E501
    coordinates = [(1.0, 0.3333333333333333), (2.0, 0.6666666666666666)]
    manager = CoordinateManager(coordinates)
    assert len(manager.points) == 2
    for point in manager.points:
        assert 0 <= point.y <= 1


def test_coordinate_manager_boundary_y_values():
    """
    Test that CoordinateManager accepts y-coordinates exactly at 0 and 1.
    """  # noqa E501
    coordinates = [(1.0, 0.0), (2.0, 0.5), (3.0, 1.0)]
    manager = CoordinateManager(coordinates)
    assert len(manager.points) == 3
