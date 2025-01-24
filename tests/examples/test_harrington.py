from pumas.data.datasets import harrington_dataset


def test_harrington_has_data():
    assert harrington_dataset.data is not None


def test_harrington_has_data_frame():
    assert harrington_dataset.data_frame is not None


def test_harrington_has_description():
    assert harrington_dataset.description is not None
