from typing import Callable, Iterable
from typing import Any


def apply_to_leaves(
        data: Iterable,
        func: callable,
        on_exception: Callable[[Iterable, Exception], Any] = None
) -> Iterable:
    """
    Recursively applies a function to each leaf value in a data structure.

    Parameters:
    - data: The input data structure (could be a dict, list, tuple, set, or a scalar value).
    - func: A function to apply to each leaf value.

    Returns:
    - The modified data structure with the function applied to each leaf value.
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = apply_to_leaves(data[key], func, on_exception)
        return data
    elif isinstance(data, list):
        for index in range(len(data)):
            data[index] = apply_to_leaves(data[index], func, on_exception)
        return data
    elif isinstance(data, tuple):
        return tuple(apply_to_leaves(item, func, on_exception) for item in data)
    elif isinstance(data, set):
        return set(apply_to_leaves(item, func, on_exception) for item in data)
    else:
        try:
            return func(data)
        except Exception as e:
            if on_exception:
                return on_exception(data, e)
            else:
                raise
