from pumas.aggregation import aggregation_catalogue
from pumas.architecture.catalogue import Catalogue


def test_desirability_catalogue_is_accessible():
    assert aggregation_catalogue
    assert isinstance(aggregation_catalogue, Catalogue)


def check_name_key_match():
    """Utility function to check if the item's name matches the key in the catalogue."""
    maps = [
        {"name": aggregation_catalogue.get(item).name, "key": item}
        for item in aggregation_catalogue.list_items()
    ]

    assert all(map_item["name"] == map_item["key"] for map_item in maps), (
        f"Mismatch between item names and catalogue keys. "
        f"Check registrations. {maps}"
    )
