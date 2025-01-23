import pumas


def test_version():
    version = pumas.__version__

    assert version is not None

