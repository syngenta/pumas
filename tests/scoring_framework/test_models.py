from pumas.scoring_framework.models import (
    InputData,
    ObjectPropertiesMap,
    ScoringResult,
    ScoringResults,
)


def test_object_properties_map():
    """
    Test the functionality of ObjectPropertiesMap, including get, set, delete,
    length, keys, values, update, and items methods.
    """
    opm = ObjectPropertiesMap.model_validate({"a": 1, "b": 2})

    assert opm.get("a") == 1
    assert opm["b"] == 2
    assert opm.get("c") is None

    opm["c"] = 3
    assert opm["c"] == 3

    del opm["b"]
    assert "b" not in opm

    assert len(opm) == 2
    assert set(opm.keys()) == {"a", "c"}
    assert set(opm.values()) == {1, 3}

    opm.update({"d": 4})
    assert opm["d"] == 4
    assert opm.items() == {"a": 1, "c": 3, "d": 4}.items()


def test_input_data():
    """
    Test the creation and items method of InputData.
    """
    opm1 = ObjectPropertiesMap.model_validate({"obj1": 1, "obj2": 2})
    opm2 = ObjectPropertiesMap.model_validate({"obj1": 3, "obj2": 4})
    input_data = InputData(data={"item1": opm1, "item2": opm2})

    assert dict(input_data.items()) == {"item1": opm1, "item2": opm2}


def test_input_data_validate_objective_data():
    """
    Test the validate_objectives method of InputData.
    """
    opm1 = ObjectPropertiesMap.model_validate({"obj1": 1, "obj2": 2})
    opm2 = ObjectPropertiesMap.model_validate({"obj1": 3, "obj2": 4})
    input_data = InputData(data={"item1": opm1, "item2": opm2})

    assert input_data.validate_objectives(["obj1", "obj2"]) is True
    assert input_data.validate_objectives(["obj1", "obj3"]) is False


def test_scoring_result():
    """
    Test the creation and attribute access of ScoringResult.
    """
    sr = ScoringResult(
        aggregated_score=0.5, desirability_scores={"obj1": 0.6, "obj2": 0.4}
    )

    assert sr.aggregated_score == 0.5
    assert sr.desirability_scores == {"obj1": 0.6, "obj2": 0.4}


def test_scoring_results():
    """
    Test the creation of ScoringResults using from_dicts method,
    as well as get_result_by_uid, item access, and items methods.
    """

    data = {
        "results": {
            "item1": {
                "aggregated_score": 0.5,
                "desirability_scores": {"obj1": 0.6, "obj2": 0.4},
            },
            "item2": {
                "aggregated_score": 0.6,
                "desirability_scores": {"obj1": 0.7, "obj2": 0.5},
            },
        }
    }
    sr = ScoringResults.model_validate(data)

    assert sr.get_result_by_uid("item1").aggregated_score == 0.5
    assert sr.get_result_by_uid("item2").desirability_scores == {
        "obj1": 0.7,
        "obj2": 0.5,
    }

    assert sr["item1"].aggregated_score == 0.5

    items = dict(sr.items())
    assert set(items.keys()) == {"item1", "item2"}
    assert all(isinstance(v, ScoringResult) for v in items.values())


def test_object_properties_map_initialization():
    """
    Test the initialization of ObjectPropertiesMap using model_validate.
    """
    opm2 = ObjectPropertiesMap.model_validate({"c": 3, "d": 4})
    assert opm2.data == {"c": 3, "d": 4}
