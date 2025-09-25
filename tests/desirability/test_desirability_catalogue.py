from pumas.architecture.catalogue import Catalogue
from pumas.desirability import desirability_catalogue


def test_desirability_catalogue_is_accessible():
    assert desirability_catalogue
    assert isinstance(desirability_catalogue, Catalogue)


def check_name_key_match():
    """Utility function to check if the item's name matches the key in the catalogue."""
    maps = [
        {"name": desirability_catalogue.get(item).name, "key": item}
        for item in desirability_catalogue.list_items()
    ]

    assert all(map_item["name"] == map_item["key"] for map_item in maps), (
        f"Mismatch between item names and catalogue keys. "
        f"Check registrations. {maps}"
    )
