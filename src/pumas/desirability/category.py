import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel, field_validator

from pumas.desirability.base_models import Desirability
from pumas.uncertainty.uncertainties_wrapper import UFloat


class CategoryPoint(BaseModel):
    category: str
    value: float

    @field_validator("value")
    def validate_value(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0 and 1")
        return v

    def __hash__(self):
        return hash((self.category, self.value))

    def __eq__(self, other):
        if not isinstance(other, CategoryPoint):
            return NotImplemented
        return self.category == other.category and math.isclose(self.value, other.value)


def check_empty_list(categories: List[Tuple[str, float]]) -> None:
    if not categories:
        raise ValueError("Categories list cannot be empty.")


def check_single_category(categories: List[Tuple[str, float]]) -> None:
    if len(categories) == 1:
        raise ValueError("At least two categories are required.")


def build_category_points(
    categories: List[Tuple[str, float]]
) -> Tuple[Set[CategoryPoint], List[Tuple[str, float]]]:
    points = set()
    failed_categories = []

    for category, value in categories:
        try:
            point = CategoryPoint(category=category, value=value)
            points.add(point)
        except (ValueError, TypeError):
            failed_categories.append((category, value))

    return points, failed_categories


def check_duplicate_categories(points: Set[CategoryPoint]) -> None:
    categories = [point.category for point in points]
    duplicate_categories = [
        cat for cat, count in Counter(categories).items() if count > 1
    ]
    if duplicate_categories:
        raise ValueError(
            f"Duplicate categories found: {', '.join(duplicate_categories)}"
        )


class CategoryManager:
    def __init__(self, categories: List[Tuple[str, float]]):
        check_empty_list(categories)
        check_single_category(categories)

        points, failed_categories = build_category_points(categories)

        if failed_categories:
            raise ValueError(
                f"Error converting categories to CategoryPoint: "
                f"{', '.join(str(cat) for cat in failed_categories)}"
            )

        check_duplicate_categories(points)
        self.points = points


def category(x: str, categories: Iterable[Tuple[str, float]], shift: float) -> float:
    _ = CategoryManager(categories=list(categories))

    category_dict = {cat: val for cat, val in categories}
    if x not in category_dict:
        raise ValueError(f"Category '{x}' not found in provided categories")

    result = category_dict[x]

    # Apply the shift
    result = result * (1 - shift) + shift

    return result


class CategoryDesirability(Desirability):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._set_parameter_definitions(
            {
                "categories": {"type": "iterable", "default": None},
                "shift": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.0},
            }
        )
        self._validate_and_set_parameters(params)

    def compute_string(self, x: str) -> float:
        """
        Compute the category desirability for a string input.

        Args:
            x (float): The input value.

        Returns:
            float: The computed desirability value.

        Raises:
            InvalidParameterTypeError: If the input is not a float.
            ParameterValueNotSet: If any required parameter is not set.
        """
        self._validate_input(x, str)
        self._check_coefficient_parameters_values()
        parameters = self.get_parameters_values()
        return category(x=x, **parameters)

    def compute_numeric(self, x: float) -> float:
        raise NotImplementedError

    def compute_ufloat(self, x: UFloat) -> UFloat:
        raise NotImplementedError

    __call__ = compute_string
