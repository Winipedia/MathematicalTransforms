from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.laplace import LaplaceTransform

from sympy import exp, sin, Heaviside, sqrt
from sympy.abc import s, a, t, omega


class TestLaplaceTransform(BaseTestTransform):

    _transform_class = LaplaceTransform

    _transform_function_to_solution_dict = {
        Heaviside(t): 1/s,
        t*Heaviside(t): s**(-2),
        exp(-a*t)*Heaviside(t): 1/(a + s),
        omega*sin(t*sqrt(omega**2))*Heaviside(t)/sqrt(omega**2): omega/(omega**2 + s**2)
    }

    values_str = "values"
    time_points_str = "time_points"
    s_values_str = "s_values"

    _transform_data_kwargs_to_solution = (
        # Delta Function (Impulse Signal)
        (
            {
                values_str: [1, 0, 0, 0],
                time_points_str: [0, 1, 2, 3],
                s_values_str: [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j],
            },
            [1, 1, 1, 1],  # Flat spectrum
        ),
    )
