from typing import Any, Iterable, Sequence, Mapping, Callable

import numpy as np
from mpmath import almosteq
from sympy import simplify, Expr, Eq
import sympy as sp
from z3 import Solver, unsat

from utils.consts import ZERO
from utils.util import apply_to_leaves


def to_number(number: Any) -> sp.Number:
    if hasattr(number, 'has') and number.has(sp.pi):
        return number  # Keep symbolic if pi is present
    return sp.N(number)


def apply_to_number_to_leaves(
        values: Iterable,
        on_exception: Callable[[Iterable, Exception], Any] = None
) -> Any:
    return apply_to_leaves(values, to_number, on_exception)


def apply_to_number_to_leaves_safely(values: Iterable):
    def safely(data, _):
        return data
    return apply_to_number_to_leaves(values=values, on_exception=safely)


def mean(numbers):
    m = sum(sp.Rational(num) for num in numbers) / len(numbers)
    m = to_number(m)
    return m


def generate_standard_values_for_axis(values):
    return apply_to_number_to_leaves(list(range(len(values))))


def generate_deltas_for_summation(values: Sequence):
    deltas = [values[i+1] - values[i] for i in range(len(values)-1)]
    # add an extra delta t based on average to allow usage of all f_values in the summation
    deltas.append(mean(deltas))
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

    a, b = to_number(a), to_number(b)
    return almost_equal_to_decimal_places(a, b, dps_tol=dps_tol)


def subs_zero(func):
    return func.subs({ZERO: 0})


def functions_are_equal(f1, f2):
    if funcs_are_equal_sympy(f1, f2):
        return True
    elif funcs_are_equal_z3(f1, f2):
        return True
    if funcs_are_equal_number_test(f1, f2):
        return True
    else:
        return False


def funcs_are_equal_z3(f1, f2):
    solver = Solver()
    solver.add(f1 != f2)

    are_equal = solver.check() == unsat

    return are_equal


def funcs_are_equal_sympy(f1, f2):
    """
    Check if two SymPy expressions or functions are equivalent.

    Parameters:
    - f1, f2: The SymPy expressions or functions to compare.
    - symbols: A list of SymPy symbols used in the expressions (optional, for numeric checks).
    - test_values: A dictionary of specific values to substitute into the expressions (optional).
    - tolerance: Numerical tolerance for comparing numeric evaluations.

    Returns:
    - Boolean indicating whether the functions are equivalent.
    """

    # Check equivalence via simplification
    if simplify(f1 - f2) == 0:
        return True

    # Check equivalence via explicit equality (Eq)
    e = Eq(f1, f2)
    if isinstance(e, bool) and e:
        return True

    # Check structural equality directly
    if f1 == f2:
        return True

    try:
        if f1.equals(f2):
            return True
    except TypeError:
        return False

    return False


def funcs_are_equal_number_test(expr1, expr2, num_tests=1, tolerance=1e-9, retry=0):
    """
    Test if two SymPy expressions are equivalent by substituting random values
    and evaluating numerically using evalf.

    Parameters:
    - expr1, expr2: SymPy expressions to compare.
    - num_tests: Number of random test points to evaluate.
    - tolerance: Numeric tolerance for comparing floating-point results.

    Returns:
    - True if the functions are equivalent within the test range, False otherwise.
    """
    # Dynamically find the symbols in the expressions
    symbols_set1 = expr1.free_symbols
    symbols_set2 = expr2.free_symbols
    symbols_set = symbols_set1 | symbols_set2

    # Generate random test points and evaluate
    for _ in range(num_tests):
        # Generate random values for each symbol in [-10, 10]
        test_values = {sym: np.random.randint(-10, 10) for sym in symbols_set}
        test_values_1 = {sym: val for sym, val in test_values.items() if sym in symbols_set1}
        test_values_2 = {sym: val for sym, val in test_values.items() if sym in symbols_set2}

        # Evaluate the expressions using evalf
        val1 = expr1.subs(test_values_1).evalf()
        val2 = expr2.subs(test_values_2).evalf()

        if val1 == val2:
            continue

        if isinstance(val1, float) or isinstance(val2, float):
            if np.isnan(float(val1)) or np.isnan(float(val2)):
                if retry < 5:
                    return funcs_are_equal_number_test(expr1, expr2, retry=retry+1)

        # the following is done to convert any special notations like mantissa to comparable floats again
        val1_r, val1_i = val1.as_real_imag()
        val2_r, val2_i = val2.as_real_imag()

        val1_r, val1_i = sp.Float(str(val1_r)), sp.Float(str(val1_i))
        val2_r, val2_i = sp.Float(str(val2_r)), sp.Float(str(val2_i))

        val1 = val1_r + val1_i * sp.I
        val2 = val2_r + val2_i * sp.I

        # Compare results
        if not almosteq(val1, val2, abs_eps=tolerance):
            return False

    # All tests passed
    return True


def evaluate_function(function: Expr, subs: dict, **kwargs) -> sp.N:
    subs[ZERO] = 0
    evaluated = function.subs(subs).evalf(**kwargs)
    evaluated = to_number(evaluated)
    return evaluated


def replace_unevaled_integrals_with_forms(expr: sp.Expr):

    expr = replace_integral_with_dirac_delta(expr)

    return expr


def replace_integral_with_dirac_delta(expr):
    # Traverse the expression to find all Integrals
    for integral in expr.atoms(sp.Integral):
        limit_var = integral.limits[0][0]
        integrand = integral.function
        # Match exponential terms of the form exp(I*k*variable)
        k = sp.Wild('k')
        c = sp.Wild('c')
        exp_match = integrand.match(c * sp.exp(sp.I * k * limit_var))

        if exp_match:
            k_val = exp_match[k]
            c_val = exp_match[c]
            if limit_var in c_val.free_symbols:
                continue
            # Replace integral with Dirac delta
            dirac_delta = c_val * ((2 * sp.pi) * sp.DiracDelta(k_val))
            dirac_delta = simplify_dirac_delta(dirac_delta)
            expr = expr.subs(integral, dirac_delta)

    return expr


def simplify_dirac_delta(delta_expr):
    if isinstance(delta_expr, sp.Mul):
        # Identify factors involving DiracDelta
        for term in delta_expr.atoms(sp.DiracDelta):
            arg = term.args[0]
            # Check for scaling in the Dirac delta argument
            if arg.is_Mul:
                variable = list(arg.free_symbols)[0]
                scaling_factor = arg / variable
                if variable.is_Symbol:
                    # Apply delta(a*x) = (1/|a|) delta(x)
                    new_delta = sp.DiracDelta(variable) / sp.Abs(scaling_factor)
                    return delta_expr / term * new_delta
    return delta_expr


def list_to_numbers(numbers: list):

    return list(map(to_number, numbers))
