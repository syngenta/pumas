import math
from collections import Counter
from typing import Iterable, List, Set, Tuple, Union

from pydantic import BaseModel, field_validator

from pumas.desirability.base_models import Desirability


class Point(BaseModel):
    x: Union[int, float]
    y: Union[int, float]

    @field_validator("x", "y")
    def validate_finite(cls, v):
        if not isinstance(v, (int, float)) or (
            isinstance(v, float) and (math.isnan(v) or math.isinf(v))
        ):
            raise ValueError("Coordinates must be finite numbers")
        return v

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)


def check_empty_input_coordinates(coordinates: List[Tuple[float, float]]) -> None:
    if not coordinates:
        raise ValueError("Coordinates list cannot be empty.")


def check_single_input_coordinate(coordinates: List[Tuple[float, float]]) -> None:
    if len(coordinates) == 1:
        raise ValueError(
            "At least two coordinates are required to form a valid multistep."
        )


def build_points(
    coordinates: List[Tuple[float, float]]
) -> Tuple[Set[Point], List[Tuple[float, float]]]:
    points = set()
    failed_coordinates = []

    for x, y in coordinates:
        try:
            point = Point(x=x, y=y)
            points.add(point)
        except (ValueError, TypeError):
            failed_coordinates.append((x, y))

    return points, failed_coordinates


def check_duplicate_x_coordinates(points: Set[Point]) -> None:
    x_coords = [point.x for point in points]
    duplicate_x_coords = [x for x, count in Counter(x_coords).items() if count > 1]
    if duplicate_x_coords:
        raise ValueError("Duplicate x-coordinates found.")


def check_boundaries_y_coordinates(points: Set[Point]) -> None:
    out_of_bounds = [point for point in points if not 0 <= point.y <= 1]
    if out_of_bounds:
        raise ValueError(
            "Y-coordinate must be between 0 and 1."
            "Please review the following coordinates: "
            ", ".join(str(coord) for coord in out_of_bounds)
        )


def check_failures(failed_coordinates: List[Tuple[float, float]]) -> None:
    if failed_coordinates:
        raise ValueError(
            f"Error converting "
            f"coordinates to Point: "
            f"{', '.join(str(coord) for coord in failed_coordinates)}"
        )


def check_single_coordinate(points: Set[Point]) -> None:
    if len(points) == 1:
        raise ValueError(
            "At least two DIFFERENT coordinates are required to form a valid multistep."
        )


def sort_points(points: Set[Point]) -> List[Point]:
    return sorted(points, key=lambda p: p.x)


class CoordinateManager:
    """
    points is a list of Point objects sorted by x-coordinate.
    """

    def __init__(self, coordinates: List[Tuple[float, float]]):
        check_empty_input_coordinates(coordinates=coordinates)
        check_single_input_coordinate(coordinates=coordinates)

        points, failed_coordinates = build_points(coordinates)

        check_failures(failed_coordinates=failed_coordinates)

        check_single_coordinate(points=points)
        check_duplicate_x_coordinates(points=points)
        check_boundaries_y_coordinates(points=points)

        self.points = sort_points(points=points)


def interpolate(x: float, p1: Point, p2: Point) -> float:
    t = (x - p1.x) / (p2.x - p1.x)
    return p1.y + t * (p2.y - p1.y)


def multistep(
    x: float,
    coordinates: Iterable[Tuple[Union[int, float], Union[int, float]]],
    shift: float = 0.0,
) -> float:
    if not 0 <= shift <= 1:
        raise ValueError("Shift must be between 0 and 1")

    cm = CoordinateManager(coordinates=list(coordinates))
    points = cm.points

    result = None
    if x <= points[0].x:
        result = points[0].y
    elif x >= points[-1].x:
        result = points[-1].y
    else:
        for i in range(len(points) - 1):
            if points[i].x <= x <= points[i + 1].x:
                result = interpolate(x, points[i], points[i + 1])
                break

    if result is None:
        raise ValueError(f"Unable to interpolate for x={x}")

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class MultiStep(Desirability):
    def __init__(self):
        super().__init__(
            utility_function=multistep,
            coefficient_parameters_names=["coordinates", "shift"],
            input_parameters_names=["x"],
        )
        # Set defaults and boundaries for the parameters:
        attributes_change_map = {
            "coordinates": {"default": None},
            "shift": {"min": 0.0, "max": 1.0, "default": 0.0},
        }
        self.set_coefficient_parameters_attributes(attributes_map=attributes_change_map)
