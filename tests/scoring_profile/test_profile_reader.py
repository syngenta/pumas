import json
from pathlib import Path

from pumas.scoring_profile.models import Profile
from pumas.scoring_profile.reader import read_profile


def test_read_profile(tmpdir):
    # Create a sample profile JSON file in tmpdir
    sample_profile = {
        "objectives": [
            {
                "name": "prop1",
                "desirability_function": {
                    "name": "function1",
                    "parameters": {"param1": 1},
                },
                "weight": 0.5,
                "value_type": "float",
                "kind": "numerical",
            }
        ],
        "aggregation_function": {"name": "aggregate1", "parameters": {"param1": 2}},
    }
    file_path = Path(tmpdir) / "profile.json"
    with file_path.open("w") as file:
        json.dump(sample_profile, file)

    # Read the profile using the function
    profile = read_profile(file_path)

    # Assertions to validate the profile content
    assert isinstance(profile, Profile)
    assert profile.objectives[0].name == "prop1"
