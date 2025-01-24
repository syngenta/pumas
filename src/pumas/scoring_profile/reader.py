import json
from pathlib import Path

from pumas.scoring_profile.models import Profile


def read_profile(file_path: Path) -> Profile:
    """
    Reads a scoring profile from a JSON file and returns a ProfileModel instance.
    """
    with file_path.open("r") as file:
        profile_data = json.load(file)
    return Profile(**profile_data)
