import pytest

from pumas.desirability import desirability_catalogue


@pytest.fixture
def desirability():
    desirability_class = desirability_catalogue.get("value_mapping")
    desirability_instance = desirability_class()
    return desirability_instance


def test_multistep_results_compute_score(desirability):
    desirability.set_coefficient_parameters_values(
        {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.0}
    )
    result = desirability.compute_score(x="Low")
    assert result == pytest.approx(expected=0.2)


# let this test fail
@pytest.mark.xfail
def test_multistep_results_compute_uscore(desirability):
    desirability.set_coefficient_parameters_values(
        {"mapping": {"Low": 0.2, "Medium": 0.5, "High": 0.8}, "shift": 0.0}
    )
    result = desirability.compute_uscore(x="Low")
    assert result == pytest.approx(expected=0.2)
