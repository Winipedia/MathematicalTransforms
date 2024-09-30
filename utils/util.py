from collections.abc import Iterable


def apply_to_leaves(data: Iterable, func: callable) -> Iterable:
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
            data[key] = apply_to_leaves(data[key], func)
        return data
    elif isinstance(data, list):
        for index in range(len(data)):
            data[index] = apply_to_leaves(data[index], func)
        return data
    elif isinstance(data, tuple):
        return tuple(apply_to_leaves(item, func) for item in data)
    elif isinstance(data, set):
        return set(apply_to_leaves(item, func) for item in data)
    else:
        return func(data)


def is_strictly_ascending(iterable: Iterable):
    return all(x < y for x, y in zip(iterable, iterable[1:]))
