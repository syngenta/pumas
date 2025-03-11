import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, field_validator

from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat


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
    """Check if the input coordinates list is empty."""
    if not coordinates:
        raise ValueError("Coordinates list cannot be empty.")


def check_single_input_coordinate(coordinates: List[Tuple[float, float]]) -> None:
    """Check if there's only one input coordinate."""
    if len(coordinates) == 1:
        raise ValueError(
            "At least two coordinates are required to form a valid multistep."
        )


def build_points(
    coordinates: List[Tuple[float, float]]
) -> Tuple[Set[Point], List[Tuple[float, float]]]:
    """
    Convert input coordinates to Point objects.

    Returns:
        Tuple[Set[Point], List[Tuple[float, float]]]:
            A set of valid Points and a list of failed coordinates.
    """
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
    """Check for duplicate x-coordinates in the set of Points."""
    x_coords = [point.x for point in points]
    duplicate_x_coords = [x for x, count in Counter(x_coords).items() if count > 1]
    if duplicate_x_coords:
        raise ValueError("Duplicate x-coordinates found.")


def check_boundaries_y_coordinates(points: Set[Point]) -> None:
    """Check if all y-coordinates are within the [0, 1] range."""

    out_of_bounds = [point for point in points if not 0 <= point.y <= 1]
    if out_of_bounds:
        raise ValueError(
            "Y-coordinate must be between 0 and 1."
            "Please review the following coordinates: "
            ", ".join(str(coord) for coord in out_of_bounds)
        )


def check_failures(failed_coordinates: List[Tuple[float, float]]) -> None:
    """Raise an error if there were any failed coordinate conversions."""
    if failed_coordinates:
        raise ValueError(
            f"Error converting "
            f"coordinates to Point: "
            f"{', '.join(str(coord) for coord in failed_coordinates)}"
        )


def check_single_coordinate(points: Set[Point]) -> None:
    """Check if there's only one unique coordinate after conversion."""
    if len(points) == 1:
        raise ValueError(
            "At least two DIFFERENT coordinates are required to form a valid multistep."
        )


def sort_points(points: Set[Point]) -> List[Point]:
    """Sort Points by their x-coordinate."""
    return sorted(points, key=lambda p: p.x)


class CoordinateManager:
    """
    Manages the coordinates for the MultiStep desirability function.

    This class handles the validation and preparation of input coordinates.

    Attributes:
        points (List[Point]): A list of Point objects sorted by x-coordinate.
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


def multistep(
    x: Union[float, UFloat],
    coordinates: Iterable[Tuple[float, float]],
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

    Raises:
        ValueError: If interpolation fails.
    """  # noqa: E501

    cm = CoordinateManager(coordinates=list(coordinates))
    points = cm.points

    result = None
    if x <= points[0].x:  # type: ignore  # this might not work with ufloat
        result = points[0].y
    elif x >= points[-1].x:  # type: ignore  # this might not work with ufloat
        result = points[-1].y
    else:
        for i in range(len(points) - 1):
            if points[i].x <= x <= points[i + 1].x:  # type: ignore  # this might not work with ufloat # noqa: E501
                result = interpolate(x, points[i], points[i + 1])  # type: ignore # review usage of type int in Point  # noqa: E501
                break

    if result is None:
        raise ValueError(f"Unable to interpolate for x={x}")

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


compute_numeric_multistep = multistep
compute_ufloat_multistep = multistep


class MultiStep(Desirability):
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

        Attributes:
            coordinates (Iterable[Tuple[float, float]]):
                The coordinates defining the multistep function. Each tuple contains (x, y) values.
            shift (float): Vertical shift of the function, range [0.0, 1.0], default 0.0.


        Raises:
            ValueError: If the coordinates list is empty.
            ValueError: If only one coordinate is provided (at least two are required).
            ValueError: If duplicate x-coordinates are found.
            ValueError: If any y-coordinate is not between 0 and 1.
            ValueError: If any coordinate cannot be converted to a
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

    def compute_numeric(self, x: float) -> float:
        """
        Compute the multistep desirability for a numeric input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, float)
        self._check_coefficient_parameters_values()
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
        self._validate_input(x, UFloat)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return compute_ufloat_multistep(x=x, **parameters)  # type: ignore

    __call__ = compute_numeric
