import random
import re
from decimal import Decimal
from functools import wraps
from typing import Union, Any, Iterable, Sequence, Mapping, Callable

from mpmath import mpc, mp, almosteq
from sympy import simplify, Expr
import sympy as sp

from utils.consts import ZERO
from utils.util import apply_to_leaves

TYPE_ANY_NUMBER = Union[str, int, float, complex, Decimal, mpc]


def get_precision():
    return mp.dps


def set_precision(new_precision):
    precision_before = get_precision()
    if precision_before != new_precision:
        mp.dps = new_precision
        after = get_precision()
        assert after == new_precision


def to_mpc(
    number: TYPE_ANY_NUMBER
) -> mpc:
    """
    Convert a given number into a mpc instance.

    :param number: The input number (can be str, int, float, Decimal, complex, mpc, or ComplexNumber)
    :return: A mpc instance
    """

    if isinstance(number, mpc):
        # Convert mpc to ComplexNumber
        return number

    elif isinstance(number, (float, Decimal, complex, str)):
        # Convert the number to a string and pass it to ComplexNumber
        number = complex(str(number))
        real, imaginary = str(number.real), str(number.imag)
        return mpc(real, imaginary)

    elif isinstance(number, sp.Add):
        if is_effectively_zero_regex(expr=number):
            return 0

    return mpc(number)


def apply_to_mpc_to_leaves(
        values: Iterable,
        on_exception: Callable[[Iterable, Exception], Any] = None
) -> Any:
    return apply_to_leaves(values, to_mpc, on_exception)


def apply_to_mpc_to_leaves_safely(values: Iterable):
    def safely(data, _):
        return data
    return apply_to_mpc_to_leaves(values=values, on_exception=safely)


def mean_mpc(iterable):
    sum_ = mp.fsum(iterable)
    mean_ = sum_ / len(iterable)
    return mean_


def generate_standard_values_for_axis(values):
    return apply_to_mpc_to_leaves(list(range(len(values))))


def generate_deltas_for_summation(values: Sequence[mpc]):
    deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
    # add an extra delta t based on average to allow usage of all f_values in the summation
    deltas.append(mean_mpc(deltas))
    return deltas


def almost_equal_to_decimal_places(a, b, dps_tol=None):
    rel_eps = 10**(-dps_tol) if dps_tol else None
    return almosteq(a, b, rel_eps=rel_eps)


def deep_almost_equal(a, b, dps_tol=None):
    """
    Recursively checks if two data structures are almost equal.
    """

    if isinstance(a, Iterable):
        assert type(a) is type(b), "The nested structure must be of equal types in depth"
        if isinstance(a, Mapping):
            assert a.keys() == b.keys(), "Keys must be the same"
            a, b = a.values(), b.values()
        for c, d in zip(a, b):
            if not deep_almost_equal(c, d, dps_tol=dps_tol):
                return False
        return True

    a, b = to_mpc(a), to_mpc(b)
    return almost_equal_to_decimal_places(a, b, dps_tol=dps_tol)


def subs_zero(func):
    return func.subs({ZERO: 0})


def functions_are_equal(f1, f2, dps_tol=None, num_tests=1):
    """
    Compare two sympy expressions to determine if they are mathematically equal.

    Parameters:
    f1, f2 : sympy expressions
        The expressions to compare.
    variables : list of sympy symbols, optional
        The variables involved in the expressions. If None, they will be inferred.
    num_tests : int, optional
        The number of random numerical tests to perform if symbolic methods fail.
    tol : float, optional
        The numerical tolerance for comparing numerical evaluations.

    Returns:
    bool
        True if the expressions are equal, False otherwise.
    """
    f1 = subs_zero(f1)
    f2 = subs_zero(f2)
    # Try symbolic equality
    try:
        if f1.equals(f2):
            return True
    except TypeError:
        pass

    try:
        # Try simplifying the difference
        diff = simplify(f1 - f2)
        if diff == 0:
            return True
    except TypeError:
        pass

    # If symbolic methods failed, proceed to numerical testing
    variables = list(f1.free_symbols.union(f2.free_symbols))

    # Generate random test points
    for _ in range(num_tests):
        val_dict = {}
        for var in variables:
            # Generate a random value avoiding zeros to prevent division errors
            val = round(random.uniform(-10, 10), 2)
            val_dict[var] = val

        val1 = f1.subs(val_dict).evalf()
        val2 = f2.subs(val_dict).evalf()
        val1, val2 = to_mpc(val1), to_mpc(val2)
        if not almost_equal_to_decimal_places(val1, val2, dps_tol=dps_tol):
            return False

    return True


def evaluate_function(function: Expr, subs: dict, **kwargs) -> mpc:
    subs[ZERO] = 0
    evaluated = function.evalf(subs=subs, **kwargs)
    evaluated = to_mpc(evaluated)
    return evaluated


def convert_all_input_output_to_mpc_safely(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = apply_to_mpc_to_leaves_safely(args)
        kwargs = apply_to_mpc_to_leaves_safely(kwargs)
        result = func(*args, **kwargs)
        result = apply_to_mpc_to_leaves_safely(result)
        return result
    return wrapper


def is_effectively_zero_regex(expr):
    # Define a regex pattern to match numbers like -0.e+100 or 0.e-100 with exponent >= 100
    pattern = r'^-?0\.e[+-]?[1-9]\d{2,}$'

    # Convert real and imaginary parts to strings
    real_part = str(sp.re(expr))
    imag_part = str(sp.im(expr))

    # Check if both real and imaginary parts match the pattern
    real_match = re.match(pattern, real_part.strip())
    imag_match = re.match(pattern, imag_part.strip())

    # Return True if both real and imaginary parts are effectively zero with exponent >= 100
    return bool(real_match) and bool(imag_match)
