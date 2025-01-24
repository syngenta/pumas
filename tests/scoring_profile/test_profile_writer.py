import json
from pathlib import Path

from pumas.scoring_profile.models import (
    AggregationFunction,
    DesirabilityFunction,
    Objective,
    Profile,
)
from pumas.scoring_profile.writer import write_profile


def test_write_profile(tmpdir):
    # Create a sample ProfileModel instance
    property_model = Objective(
        name="prop1",
        desirability_function=DesirabilityFunction(
            name="func1", parameters={"param1": 1}
        ),
        weight=0.5,
        value_type="float",
        kind="numerical",
    )
    aggregation_model = AggregationFunction(name="aggregate1", parameters={"param1": 2})
    original_profile = Profile(
        objectives=[property_model], aggregation_function=aggregation_model
    )

    # Write the profile to a JSON file in tmpdir
    file_path = Path(tmpdir) / "profile.json"
    write_profile(original_profile, file_path)

    # Read the file and validate the content
    with file_path.open("r") as file:
        data = json.load(file)

    # Ensure that the data_frame is a dictionary
    assert isinstance(data, dict), "Loaded data_frame is not a dictionary"

    # Recreate the profile model from the loaded data_frame
    recreated_profile = Profile(**data)

    # Assertions to validate the equality of original and recreated profiles
    assert original_profile == recreated_profile
