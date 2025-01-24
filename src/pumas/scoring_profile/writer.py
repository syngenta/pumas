from pathlib import Path

from pumas.scoring_profile.models import Profile


def write_profile(profile: Profile, file_path: Path):
    """
    Writes a scoring profile to a JSON file.
    """
    with file_path.open("w") as file:
        file.write(profile.model_dump_json(indent=2))
