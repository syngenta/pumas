import concurrent.futures
from typing import Any, Callable, Iterable, List, Tuple


def parallelize(
    func: Callable[[Any], Any],
    data: Iterable[Any],
    num_jobs: int = 0,
    method: str = "threads",
) -> List[Any]:
    if num_jobs < 0:
        raise ValueError("Number of jobs must be a non-negative integer")
    if method not in ["threads", "processes"]:
        raise ValueError("Method must be one of 'threads' or 'processes'")
    if not isinstance(func, Callable):
        raise ValueError("The func argument must be callable")

    if num_jobs == 0 or method not in ["threads", "processes"]:
        return [func(item) for item in data]

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if method == "threads"
        else concurrent.futures.ProcessPoolExecutor
    )

    try:
        with executor_cls(max_workers=num_jobs) as executor:
            results = list(executor.map(func, data))
        return results
    except Exception as e:
        raise RuntimeError(f"Parallel execution failed: {e}")


def parallelize_with_indices(
    func: Callable[[Any], Any],
    data: List[Tuple[int, Any]],
    num_jobs: int = 0,
    method: str = "threads",
) -> List[Tuple[int, Any]]:
    if num_jobs < 0:
        raise ValueError("Number of jobs must be a non-negative integer")
    if method not in ["threads", "processes"]:
        raise ValueError("Method must be one of 'threads' or 'processes'")
    if not isinstance(func, Callable):
        raise ValueError("The func argument must be callable")

    if num_jobs == 0 or method not in ["threads", "processes"]:
        return [(i, func(item)) for i, item in data]

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if method == "threads"
        else concurrent.futures.ProcessPoolExecutor
    )

    try:
        with executor_cls(max_workers=num_jobs) as executor:
            results = list(
                executor.map(_apply_func_with_index, [(func, pair) for pair in data])
            )

        # want to return the results in the order of the original data
        # this might be unnecessary if the order is guaranteed
        # to be preserved by the executor
        # we keep it here to be safe should we change the parallelization approach
        # Create a dictionary to map indices to their results
        result_dict = {index: result for index, result in results}

        # Reconstruct the list in the original order provided by the data
        ordered_results = [(index, result_dict[index]) for index, _ in data]

        return ordered_results
    except Exception as e:
        raise RuntimeError(f"Parallel execution failed: {e}")


def _apply_func_with_index(
    args: Tuple[Callable[[Any], Any], Tuple[int, Any]]
) -> Tuple[int, Any]:
    """
    NOTE to the developer: the pickle module is used to serialize the function and its arguments
    it would fail on some OS architectures if the function is not defined in the global scope of the module.
    This is why this function is defined in the global scope of the module.
    At the cost of decreased readability, this function is defined here to avoid the serialization issue with the picle module witout adding an additional dependency.
    For more information, see:
    https://stackoverflow.com/questions/72766345/attributeerror-cant-pickle-local-object-in-multiprocessing
    """  # noqa: E501
    func, (index, value) = args
    return index, func(value)
