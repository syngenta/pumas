import pytest
from pydantic import ValidationError

from pumas.scoring_profile.scoring_profile import (
    AggregationFunction,
    DesirabilityFunction,
    Objective,
    ScoringProfile,
)


@pytest.fixture()
def valid_desirability_function():
    return DesirabilityFunction(
        name="sigmoid",
        parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
    )


@pytest.fixture
def valid_objective():
    return Objective(
        name="obj1",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
        weight=0.5,
    )


@pytest.fixture
def valid_aggregation():
    return AggregationFunction(name="geometric_mean")


@pytest.fixture
def valid_profile(valid_objective, valid_aggregation):
    return ScoringProfile(
        objectives=[valid_objective], aggregation_function=valid_aggregation
    )


# Desirability Function Tests
def test_valid_desirability_function():
    """
    Test that a valid desirability function is accepted by the Objective class.
    """ ""
    DesirabilityFunction(
        name="sigmoid",
        parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
    )


def test_invalid_desirability_function():
    """Test that a desirability function that is not registered in the catalog
    is not accepted by the Objective class.
    """
    with pytest.raises(ValidationError):
        _ = DesirabilityFunction(
            name="invalid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        )


def test_desirability_function_with_invalid_parameters_not_existing_parameter_name():
    """Test that a desirability function with invalid parameters
    is not accepted by the Objective class.
    Here the case is an unexpected parameter.
    """
    with pytest.raises(ValidationError):
        _ = DesirabilityFunction(
            name="sigmoid",
            parameters={"invalid": 0.0},
        )


def test_desirability_function_with_invalid_parameters_values_none():
    """Test that a desirability function with invalid parameters
    is not accepted by the Objective class.
    Here the case is missing a mandatory parameter that has no default
    """
    with pytest.raises(ValidationError):
        _ = DesirabilityFunction(
            name="invalid",
            parameters={
                "low": 0.0,
                # "high": 1.0, # that is mandatory and has no default
                "k": 0.1,
                "base": 10.0,
                "shift": 0.0,
            },
        )


def test_desirability_function_with_invalid_parameters_value_out_of_boundary():
    """Test that a desirability function with invalid parameters
    is not accepted by the Objective class.
    Here the case is setting a parameter value out if its boundary
    """
    with pytest.raises(ValidationError):
        _ = DesirabilityFunction(
            name="invalid",
            parameters={
                "low": 0.0,
                "high": 1.0,
                "k": 0.1,
                "base": 10.0,
                "shift": 2.0,  # boundary is between 0 and 1
            },
        )


def test_desirability_function_with_invalid_parameters_value_wrong_type():
    """Test that a desirability function with invalid parameters
    is not accepted by the Objective class.
    Here the case is setting a parameter value of the wrong type
    """
    with pytest.raises(ValidationError):
        _ = DesirabilityFunction(
            name="invalid",
            parameters={
                "low": 0.0,
                "high": 1,  # it should be a float not an int
                "k": 0.1,
                "base": 10.0,
                "shift": 2.0,
            },
        )


# Aggregation Function Tests
def test_valid_aggregation_function():
    """Test that a valid aggregation function
    is accepted by the AggregationFunction class"""
    af = AggregationFunction(name="geometric_mean")
    assert af.name == "geometric_mean"


def test_invalid_aggregation_function():
    """Test that an aggregation function
    that is not registered in the catalog
    is not accepted by the Objective class.
    """
    with pytest.raises(ValidationError):
        _ = AggregationFunction(name="invalid_function")


# Objective Tests
def test_valid_objective(valid_objective):
    """Test behavior with a valid objective"""
    assert valid_objective


# Profile Tests
def test_valid_profile(valid_profile):
    """test behaviour with a valid profile"""
    assert valid_profile


def test_profile_unique_objective_names(valid_objective, valid_aggregation):
    """Test duplicated objective names raise error"""
    with pytest.raises(ValidationError):
        ScoringProfile(
            objectives=[valid_objective, valid_objective],
            aggregation_function=valid_aggregation,
        )


def test_profile_all_weights_is_ok(valid_aggregation):
    obj1 = Objective(
        name="obj1",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
        weight=0.5,
    )
    obj2 = Objective(
        name="obj2",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
        weight=0.5,
    )
    ScoringProfile(objectives=[obj1, obj2], aggregation_function=valid_aggregation)


def test_profile_non_weights_is_ok(valid_aggregation):
    obj1 = Objective(
        name="obj1",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
    )
    obj2 = Objective(
        name="obj2",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
    )
    ScoringProfile(objectives=[obj1, obj2], aggregation_function=valid_aggregation)


def test_profile_some_weights_raises(valid_objective, valid_aggregation):
    obj1 = Objective(
        name="obj1",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
    )
    obj2 = Objective(
        name="obj2",
        desirability_function=DesirabilityFunction(
            name="sigmoid",
            parameters={"low": 0.0, "high": 1.0, "k": 0.1, "base": 10.0, "shift": 0.0},
        ),
        weight=1.0,
    )
    with pytest.raises(ValidationError):
        _ = ScoringProfile(
            objectives=[obj1, obj2], aggregation_function=valid_aggregation
        )


def test_profile_serialization(valid_profile):
    """Test serialization of profile"""
    serialized = valid_profile.model_dump_json()
    assert serialized
    assert isinstance(serialized, str)
    deserialized = ScoringProfile.model_validate_json(serialized)
    assert deserialized
    assert isinstance(deserialized, ScoringProfile)


def test_profile_write_read(valid_profile, tmpdir):
    """Test file_roundtrip"""
    file_path = tmpdir / "test_profile.json"
    valid_profile.write_to_file(file_path=file_path)
    profile = ScoringProfile.read_from_file(file_path=file_path)
    assert profile == valid_profile
    assert isinstance(profile, ScoringProfile)
