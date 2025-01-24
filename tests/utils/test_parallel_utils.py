import pytest

from pumas.utils.parallel_utils import parallelize, parallelize_with_indices


def square(x):
    return x**2


def test_parallelize_serial():
    data = [1, 2, 3, 4, 5]
    result = parallelize(func=square, data=data, num_jobs=0)
    assert result == [1, 4, 9, 16, 25]


def test_parallelize_threads():
    data = [1, 2, 3, 4, 5]
    result = parallelize(func=square, data=data, num_jobs=2, method="threads")
    assert result == [1, 4, 9, 16, 25]


def test_parallelize_processes():
    data = [1, 2, 3, 4, 5]
    result = parallelize(func=square, data=data, num_jobs=2, method="processes")
    assert result == [1, 4, 9, 16, 25]


def test_parallelize_invalid_method():
    data = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        parallelize(func=square, data=data, num_jobs=2, method="invalid")


def test_parallelize_invalid_num_jobs():
    data = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        parallelize(func=square, data=data, num_jobs=-2, method="threads")


def test_parallelize_invalid_func():
    data = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        parallelize(func="not_a_function", data=data, num_jobs=2, method="threads")


def test_parallelize_execution_failure():
    def faulty_func(x):
        _ = x
        raise ValueError("Intentional error")

    data = [1, 2, 3, 4, 5]
    with pytest.raises(RuntimeError):
        parallelize(faulty_func, data, num_jobs=2, method="threads")


def test_parallelize_with_indices_serial():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    result = parallelize_with_indices(square, data, num_jobs=0)
    assert result == [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)]


def test_parallelize_with_indices_threads():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    result = parallelize_with_indices(square, data, num_jobs=2, method="threads")
    assert result == [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)]


def test_parallelize_with_indices_processes():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    result = parallelize_with_indices(square, data, num_jobs=2, method="processes")
    assert result == [(0, 1), (1, 4), (2, 9), (3, 16), (4, 25)]


def test_parallelize_with_indices_invalid_method():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    with pytest.raises(ValueError):
        parallelize_with_indices(square, data, num_jobs=2, method="invalid")


def test_parallelize_with_indices_execution_failure():
    def faulty_func(x):
        _ = x
        raise ValueError("Intentional error")

    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    with pytest.raises(RuntimeError):
        parallelize_with_indices(faulty_func, data, num_jobs=2, method="threads")


def test_parallelize_with_indices_invalid_num_jobs():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    with pytest.raises(ValueError):
        parallelize_with_indices(func=square, data=data, num_jobs=-2, method="invalid")


def test_parallelize_with_indices_invalid_func():
    data = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    with pytest.raises(ValueError):
        parallelize_with_indices(
            func="not_a_function", data=data, num_jobs=2, method="threads"
        )


@pytest.fixture
def scrambled_data():
    # this just has an index that is not in order
    return [(10, 1), (3, 2), (15, 3), (8, 4), (6, 5)]


def test_parallelize_with_indices_serial_on_scrambled_data(scrambled_data):
    result = parallelize_with_indices(square, scrambled_data, num_jobs=0)
    expected_result = [(10, 1), (3, 4), (15, 9), (8, 16), (6, 25)]
    assert result == expected_result


def test_parallelize_with_indices_threads_on_scrambled_data(scrambled_data):
    result = parallelize_with_indices(
        square, scrambled_data, num_jobs=2, method="threads"
    )
    expected_result = [(10, 1), (3, 4), (15, 9), (8, 16), (6, 25)]
    assert result == expected_result


def test_parallelize_with_indices_processes_on_scrambled_data(scrambled_data):
    result = parallelize_with_indices(
        square, scrambled_data, num_jobs=2, method="processes"
    )
    expected_result = [(10, 1), (3, 4), (15, 9), (8, 16), (6, 25)]
    assert result == expected_result
