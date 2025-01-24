from pumas.scoring_profile.models import DesirabilityFunction, Objective


def test_valid_property_model():
    # Test with valid combinations
    Objective(
        name="prop1",
        desirability_function=DesirabilityFunction(
            name="func1", parameters={"param1": 1}
        ),
        weight=0.5,
        value_type="float",
        kind="numerical",
    )

    Objective(
        name="prop2",
        desirability_function=DesirabilityFunction(
            name="func2", parameters={"param1": 2}
        ),
        weight=0.5,
        value_type="str",
        kind="categorical",
    )
