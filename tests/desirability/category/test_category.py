# type: ignore
from math import isclose

import pytest

from pumas.desirability import desirability_catalogue
from pumas.desirability.category import CategoryManager, CategoryPoint, category


def test_category_point_init():
    # Test that the CategoryPoint object is initialized correctly
    cp = CategoryPoint(category="Medium", value=0.5)
    assert cp.category == "Medium"
    assert cp.value == 0.5


def test_category_point_repr():
    # Test the string representation of the CategoryPoint object
    cp = CategoryPoint(category="High", value=0.8)
    assert repr(cp) == "CategoryPoint(category='High', value=0.8)"


def test_category_point_eq_same_point():
    # Test equality for the same CategoryPoint object
    cp1 = CategoryPoint(category="Low", value=0.2)
    cp2 = CategoryPoint(category="Low", value=0.2)
    assert cp1 == cp2


def test_category_point_eq_different_point():
    # Test equality for different CategoryPoint objects
    cp1 = CategoryPoint(category="Low", value=0.2)
    cp2 = CategoryPoint(category="Medium", value=0.5)
    assert cp1 != cp2


def test_category_point_eq_different_types():
    # Test equality with a non-CategoryPoint object
    cp = CategoryPoint(category="High", value=0.8)
    assert cp != ("High", 0.8)


def test_category_point_eq_float_precision():
    # Test equality with float precision
    cp1 = CategoryPoint(category="Medium", value=0.3333333333333333)
    cp2 = CategoryPoint(category="Medium", value=0.3333333333333334)
    assert cp1 == cp2
    assert isclose(cp1.value, cp2.value)


def test_category_manager_empty_list():
    with pytest.raises(ValueError, match="Categories list cannot be empty."):
        CategoryManager([])


def test_category_manager_single_category():
    with pytest.raises(ValueError, match="At least two categories are required."):
        CategoryManager([("Low", 0.2)])


def test_category_manager_invalid_categories():
    with pytest.raises(
        ValueError, match="Error converting categories to CategoryPoint: "
    ):
        CategoryManager([("Low", 0.2), ("Medium", "invalid"), ("High", 0.8)])
    with pytest.raises(
        ValueError, match="Error converting categories to CategoryPoint: "
    ):
        CategoryManager([("Low", 0.2), ("Medium", 1.5), ("High", 0.8)])


def test_category_manager_duplicate_categories():
    with pytest.raises(ValueError, match="Duplicate categories found: "):
        CategoryManager([("Low", 0.2), ("Medium", 0.5), ("Low", 0.3)])


def test_category_manager_valid_categories():
    categories = [("Low", 0.2), ("Medium", 0.5), ("High", 0.8)]
    manager = CategoryManager(categories)
    assert len(manager.points) == 3

    expected_points = {
        CategoryPoint(category="Low", value=0.2),
        CategoryPoint(category="Medium", value=0.5),
        CategoryPoint(category="High", value=0.8),
    }
    assert set(manager.points) == expected_points


@pytest.mark.parametrize(
    "categories_list, x, expected",
    [
        # Basic cases
        ([("Low", 0.2), ("Medium", 0.5), ("High", 0.8)], "Low", 0.2),
        ([("Low", 0.2), ("Medium", 0.5), ("High", 0.8)], "Medium", 0.5),
        ([("Low", 0.2), ("Medium", 0.5), ("High", 0.8)], "High", 0.8),
        # Case sensitivity
        ([("low", 0.2), ("medium", 0.5), ("high", 0.8)], "low", 0.2),
        # Edge cases
        ([("Extremely Low", 0.0), ("Extremely High", 1.0)], "Extremely Low", 0.0),
        ([("Extremely Low", 0.0), ("Extremely High", 1.0)], "Extremely High", 1.0),
    ],
)
def test_category_function(x, categories_list, expected):
    params = {"categories": categories_list, "shift": 0.0}

    assert category(x=x, **params) == expected


def test_category_function_invalid_input():
    categories_list = [("Low", 0.2), ("Medium", 0.5), ("High", 0.8)]
    # Test that an invalid category (not existing) raises a ValueError
    with pytest.raises(
        ValueError, match="Category 'Very High' not found in provided categories"
    ):
        category(x="Very High", categories=categories_list, shift=0.0)
    # Test that an invalid category (case mismatch) raises a ValueError
    with pytest.raises(
        ValueError, match="Category 'LOW' not found in provided categories"
    ):
        category(x="LOW", categories=categories_list, shift=0.0)


def test_category_results_compute_string():
    desirability_class = desirability_catalogue.get("category")
    params = {
        "categories": [("Low", 0.2), ("Medium", 0.5), ("High", 0.8)],
        "shift": 0.0,
    }
    desirability_instance = desirability_class(params=params)

    result = desirability_instance.compute_string(x="Medium")
    assert result == pytest.approx(expected=0.5)


def test_category_invalid_value():
    with pytest.raises(ValueError, match="Value must be between 0 and 1"):
        CategoryPoint(category="Invalid", value=1.5)
