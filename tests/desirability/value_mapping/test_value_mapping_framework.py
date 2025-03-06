import pytest

from pumas.desirability import desirability_catalogue


@pytest.fixture
def desirability_class():
    desirability_class = desirability_catalogue.get("value_mapping")
    return desirability_class


def test_value_mapping_results_compute_string(desirability_class):
    params = {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.0}
    desirability = desirability_class(params=params)
    result = desirability.compute_string(x="Low")
    assert result == pytest.approx(expected=0.2)


def test_value_mapping_results_compute_score(desirability_class):
    params = {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.0}
    desirability = desirability_class(params=params)
    with pytest.raises(NotImplementedError):
        _ = desirability.compute_numeric(x="Low")


def test_value_mapping_results_compute_uscore(desirability_class):
    params = {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.0}
    desirability = desirability_class(params=params)
    with pytest.raises(NotImplementedError):
        _ = desirability.compute_numeric(x="Low")
