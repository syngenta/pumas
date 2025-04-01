from pumas.architecture.catalogue import Catalogue
from pumas.desirability import desirability_catalogue


def test_desirability_catalogue_is_accessible():
    assert desirability_catalogue
    assert isinstance(desirability_catalogue, Catalogue)
