from sympy import exp, sqrt
from sympy.abc import r, k, a, b
from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.hankel import HankelTransform


class TestHankelTransform(BaseTestTransform):
    _transform_class = HankelTransform

    _transform_function_to_solution_dict = {
        # Symbolic transformations
        exp(-r ** 2): 1 / 2 * exp(-k ** 2 / 4),  # Gaussian function
        exp(-a * r): a/(k**3*(a**2/k**2 + 1)**(3/2)),  # Exponential decay
        1/(r*sqrt(b**2/r**2 + 1)): (exp(-b * k)) / k,  # Modified Bessel function
    }

    values_str = "values"
    r_vals_str = "r_vals"
    k_vals_str = "k_vals"
    order_str = "order"

    _transform_data_kwargs_to_solution = (
        # Delta Function (Impulse Signal)
        (
            {
                values_str: [1, 0, 0, 0],  # Delta-like signal
                r_vals_str: [0, 1, 2, 3],  # Radial points
                k_vals_str: [1, 2, 3, 4],  # k-values
                order_str: 0,  # Bessel function order
            },
            [0, 0, 0, 0],
        ),
        # Constant Function (f(r) = 1)
        (
            {
                values_str: [1, 1, 1, 1],  # Constant signal
                r_vals_str: [0, 1, 2, 3],  # Radial points
                k_vals_str: [1, 2, 3, 4],  # k-values
                order_str: 0,  # Bessel function order
            },
            [0.432823380134638, -0.118473068833468, -0.229762273948568, 0.0892197368017611],
        ),
    )

