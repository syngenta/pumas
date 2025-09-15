import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, field_validator

from pumas.desirability.base_models import Desirability
from pumas.uncertainty_management.uncertainties.uncertainties_wrapper import UFloat


class DesirabilityMultistepError(Exception):
    """Base class for exceptions in the Desirability Multistep module."""


class InvalidParameterTypeError(DesirabilityMultistepError):
    """Raised when a parameter has an invalid type."""


class InvalidCoordinateError(DesirabilityMultistepError):
    """Raised when the coordinates provided are invalid."""


class Point(BaseModel):
    """
    Represents a 2D point with x and y coordinates.

    Attributes:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
    """

    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def validate_finite(cls, v):
        if not isinstance(v, (int, float)) or (
            isinstance(v, float) and (math.isnan(v) or math.isinf(v))
        ):
            raise InvalidCoordinateError("Coordinates must be finite numbers")
        return v

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)


@dataclass
class Wall:
    """
    Represents a vertical wall in the multistep function.

    A wall consists of two points with the same x-coordinate.
    """

    x_coordinate: float
    points: List[Point]  # Always contains exactly 2 points

    @property
    def active_point(self) -> Point:
        """Return the point that should be used for evaluation at this wall."""
        return self.points[0]  # The second point in the wall is the active one


def check_empty_coordinates(coordinates: List[Tuple[float, float]]) -> None:
    """Check if coordinates list is empty."""
    if not coordinates:
        raise InvalidCoordinateError("Coordinates list cannot be empty.")


def check_insufficient_coordinates(coordinates: List[Tuple[float, float]]) -> None:
    """Check if there are enough coordinates to form a valid multistep."""
    if len(coordinates) == 1:
        raise InvalidCoordinateError(
            "At least two coordinates are required to form a valid multistep."
        )


def check_duplicate_points(points: List[Point]) -> None:
    """Check for duplicate coordinates (points with the same x and y values)."""
    seen_coords = set()
    duplicates = []

    for point in points:
        coord = (point.x, point.y)
        if coord in seen_coords:
            duplicates.append(coord)
        else:
            seen_coords.add(coord)

    if duplicates:
        raise InvalidCoordinateError(
            f"Duplicate coordinates found: "
            f"{', '.join(str(coord) for coord in duplicates)}"
        )


def check_boundaries_y_coordinates(points: List[Point]) -> None:
    """Check that all y-coordinates are between 0 and 1."""
    invalid_points = [p for p in points if not (0 <= p.y <= 1)]

    if invalid_points:
        raise InvalidCoordinateError(
            f"All y-coordinates must be between 0 and 1. "
            f"Found points with invalid y: {invalid_points}"
        )


# Point creation and organization - Open/Closed Principle
def create_points_from_input_coordinates(
    coordinates: List[Tuple[float, float]]
) -> List[Point]:
    """Create Point objects from coordinate tuples."""
    points = []
    failed_coordinates = []
    for x, y in coordinates:
        try:
            point = Point(x=x, y=y)
            points.append(point)
        except (ValueError, TypeError):
            failed_coordinates.append((x, y))
    if failed_coordinates:
        raise InvalidCoordinateError(
            f"Error converting coordinates to Point: {failed_coordinates}"
        )

    return [Point(x=x, y=y) for x, y in coordinates]


def sort_points_preserving_order(points: List[Point]) -> List[Point]:
    """
    Sort points by x-coordinate while preserving the original order of points
    with the same x-coordinate.
    """
    # Group points by x-coordinate and preserve original order
    points_by_x: Dict[float, List[Tuple[int, Point]]] = {}
    for i, point in enumerate(points):
        if point.x not in points_by_x:
            points_by_x[point.x] = []
        points_by_x[point.x].append((i, point))

    # Create sorted list of points
    sorted_x = sorted(points_by_x.keys())
    sorted_points: List[Point] = []
    for x in sorted_x:
        # Sort by original index to preserve input order
        sorted_points_at_x = sorted(points_by_x[x], key=lambda item: item[0])
        sorted_points.extend(p for _, p in sorted_points_at_x)

    return sorted_points


def build_wall_map(points: List[Point]) -> Dict[float, Wall]:
    """
    Identify walls from a list of points.

    A wall is defined as two points sharing the same x-coordinate.
    """
    # Step 1: Group points by x-coordinate while preserving original order
    points_by_x: Dict[float, List[Point]] = {}
    for point in points:
        if point.x not in points_by_x:
            points_by_x[point.x] = []
        points_by_x[point.x].append(point)

    # Step 2: Create walls from groups with exactly 2 points
    wall_map: Dict[float, Wall] = {}
    for x, points_at_x in points_by_x.items():
        if len(points_at_x) >= 2:
            wall_map[x] = Wall(x_coordinate=x, points=points_at_x)

    return wall_map


def check_wall_map(wall_map: Dict[float, Wall]) -> None:
    """Check that there are no more than 2 points with the same x-coordinate."""
    invalid_walls = {x: wall for x, wall in wall_map.items() if len(wall.points) != 2}

    if invalid_walls:
        raise InvalidCoordinateError(
            f"No more than 2 points can share the same x-coordinate. "
            f"Found {len(invalid_walls)} groups of invalid walls."
            f" Found walls: "
            f"{', '.join(f'{x}: {wall.points}' for x, wall in invalid_walls.items())}"
        )


def interpolate(x: Union[float, UFloat], p1: Point, p2: Point) -> Union[float, UFloat]:
    """
    Perform linear interpolation between two points.

    Args:
        x (Union[float, UFloat]): The x-value to interpolate.
        p1 (Point): The first point.
        p2 (Point): The second point.

    Returns:
        Union[float, UFloat]: The interpolated y-value.
    """
    t = (x - p1.x) / (p2.x - p1.x)
    return p1.y + t * (p2.y - p1.y)  # type: ignore


def evaluate_at_point(
    x: Union[float, UFloat], points: List[Point], wall_map: Dict[float, Wall]
) -> Union[float, UFloat]:
    # Check if x is at a wall
    if x in wall_map:
        return wall_map[x].active_point.y  # type: ignore  # this might not work with ufloat # noqa E501

    # Check if x is outside the x-range
    if x <= points[0].x:  # type: ignore  # this might not work with ufloat
        return points[0].y
    if x >= points[-1].x:  # type: ignore  # this might not work with ufloat
        return points[-1].y

    # Find points for interpolation
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        if p1.x <= x <= p2.x:  # type: ignore  # this might not work with ufloat
            return interpolate(x, p1, p2)

    raise ValueError(f"Unable to interpolate for x={x}")


class CoordinateManager:
    """
    Manages coordinates for the multistep function.

    This class is responsible for validating, sorting, and organizing coordinates,
    but not for evaluating the function at specific points.
    """

    def __init__(self, coordinates: List[Tuple[float, float]]):
        """Initialize with a list of coordinate tuples."""
        # Validate input
        check_empty_coordinates(coordinates)
        check_insufficient_coordinates(coordinates)

        # Create and validate points
        points = create_points_from_input_coordinates(coordinates)
        check_duplicate_points(points)
        check_boundaries_y_coordinates(points)
        points_sorted = sort_points_preserving_order(points)

        # Create and validate walls
        wall_map = build_wall_map(points_sorted)
        check_wall_map(wall_map)

        self.points = points_sorted
        self.wall_map = wall_map


def multistep(
    x: Union[float, UFloat],
    coordinates: List[Tuple[float, float]],
    shift: float = 0.0,
) -> Union[float, UFloat]:
    """
    Compute the multistep desirability value for a given input.

    Args:
        x (Union[float, UFloat]): The input value.
        coordinates (Iterable[Tuple[float, float]]): The coordinates defining the multistep function.
        shift (float, optional): Vertical shift of the function. Defaults to 0.0.

    Returns:
        Union[float, UFloat]: The computed desirability value.
    """  # noqa: E501
    coordinate_manager = CoordinateManager(coordinates=coordinates)

    wall_map = coordinate_manager.wall_map
    points = coordinate_manager.points

    # Evaluate at the given x
    result = evaluate_at_point(x=x, points=points, wall_map=wall_map)

    # Apply the shift
    return result * (1 - shift) + shift


compute_numeric_multistep = multistep
compute_ufloat_multistep = multistep


class MultiStep(Desirability):
    name = "multistep"
    """
        MultiStep desirability function implementation.

        This class implements a multistep desirability function with linear interpolation
        between defined points. It provides methods to compute the desirability for both
        numeric and uncertain float inputs.

        The multistep function is defined by a set of coordinates (x_i, y_i), where:
        - x_i represents the input values (independent variable)
        - y_i represents the corresonding desirability values (dependent variable)
        - The coordinates are ordered such that x_1 < x_2 < ... < x_n
        - Each y_i must be in the range [0, 1]

        Let (x_1, y_1) be the first point and (x_n, y_n) be the last point in the ordered set.

        The multistep desirability function D(x) is defined as follows:

        .. math::

            D(x) = \\begin{cases}
                y_1 & \\text{if } x \\leq x_1 \\\\
                y_n & \\text{if } x \\geq x_n \\\\
                y_i + \\frac{x - x_i}{x_{i+1} - x_i}(y_{i+1} - y_i) & \\text{if } x_i < x < x_{i+1}
            \\end{cases}

        Where:

        - For x ≤ x_1, the function returns the desirability of the first point (y_1)
        - For x ≥ x_n, the function returns the desirability of the last point (y_n)
        - For x_i < x < x_{i+1}, linear interpolation is performed between the two closest points

        The interpolation ensures a smooth transition between defined points while maintaining
        the step-like behavior at the specified coordinates.

                Finally, the shift is applied:

        .. math::

            D_{final}(x) = D(x) \\cdot (1 - shift) + shift


        Parameters:
            params (Optional[Dict[str, Any]]): Initial parameters for the multistep function.

        Content of params:
            coordinates (Iterable[Tuple[float, float]]):
                The coordinates defining the multistep function. Each tuple contains (x, y) values.
            shift (float): Vertical shift of the function, range [0.0, 1.0], default 0.0.


        Raises:
            InvalidCoordinateError: If the coordinates list is empty.
            InvalidCoordinateError: If only one coordinate is provided (at least two are required).
            InvalidCoordinateError: If any y-coordinate is not between 0 and 1.
            InvalidCoordinateError: If any coordinate cannot be converted to a
                valid Point object (internal representation of coordinates).



        Usage Example:

        >>> from pumas.desirability import desirability_catalogue

        >>> desirability_class = desirability_catalogue.get("multistep")

        >>> coords = [(0, 0), (1, 0.5), (4, 1)]
        >>> params = {"coordinates": coords, "shift": 0.0}
        >>> desirability = desirability_class(params=params)
        >>> print(desirability.get_parameters_values())
        {'coordinates': [(0, 0), (1, 0.5), (4, 1)], 'shift': 0.0}

        >>> result = desirability.compute_numeric(x=-1.0)
        >>> print(f"{result:.2f}")
        0.00

        >>> result = desirability.compute_numeric(x=0.5)
        >>> print(f"{result:.2f}")
        0.25

        >>> result = desirability.compute_numeric(x=2.5)
        >>> print(f"{result:.2f}")
        0.75

        >>> result = desirability.compute_numeric(x=5.0)
        >>> print(f"{result:.2f}")
        1.00
        """  # noqa: E501

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sigmoid desirability function.

        Args:
            params (Optional[Dict[str, Any]]): Initial parameters for the sigmoid function.
        """  # noqa: E501
        super().__init__()
        self._set_parameter_definitions(
            {
                "coordinates": {"type": "iterable", "default": None},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_numeric(self, x: Union[int, float]) -> float:
        """
        Compute the multistep desirability for a numeric input.

        Args:
            x (Union[int, float]): The numeric input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(item=x, expected_type=(int, float))
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_numeric_multistep(x=x, **parameters)  # type: ignore

    def compute_ufloat(self, x: UFloat) -> UFloat:
        """
        Compute the multistep desirability for an uncertain float input.

        Args:
            x (UFloat): The uncertain float input value.

        Returns:
            UFloat: The computed desirability value with uncertainty.

        Raises:
            InvalidParameterTypeError: If the input is not a UFloat.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_compute_input(x, UFloat)
        self._check_parameters_values_none()
        parameters = self.get_parameters_values()
        return compute_ufloat_multistep(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
