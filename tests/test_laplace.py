from decimal import Decimal

from mpmath import mpc

from tests.base_test.base_transform_test import TestTransform
from transforms.laplace import LaplaceTransform

from sympy import exp, sin, Heaviside, sqrt, I
from sympy.abc import s, a, t, omega

from utils.consts import ZERO
from utils.mpc import to_mpc


class TestLaplaceTransform(TestTransform):

    _transform_class = LaplaceTransform

    _transform_function_to_solution_dict = {
        Heaviside(t): 1/s,
        t*Heaviside(t): s**(-2),
        exp(-a*t)*Heaviside(t): 1/(a + s),
        omega*sin(t*sqrt(omega**2))*Heaviside(t)/sqrt(omega**2): omega/(omega**2 + s**2)
    }

    f_values_str = "f_values"
    t_values_str = "t_values"
    _transform_data_kwargs_to_solution = (
        (
            {
                f_values_str: [to_mpc(10.3), 5, 1.2, complex(-3, 2), mpc(0, -1), 1000.43],
                t_values_str: [0, 4, 5, 8, 15, 33],
            },
            41.2000000000000028421709430404007434844970703125 +
            6602.838 * exp(-33.0 * s) -
            18.0 * I * exp(-15.0 * s) +
            7.0 * (-3.0 + 2.0 * I) * exp(-8.0 * s) +
            3.6 * exp(-5.0 * s) +
            5.0 * exp(-4.0 * s)
        ),
        (
            {
                f_values_str: [0, to_mpc(complex(0.5, 7)), -3, Decimal("3.1"), 0, 2000],
                t_values_str: [0, 1, 2, 6, 10, 30]
            },
            1.0*ZERO +
            20.0*ZERO*exp(-10.0*s) +
            12000.0*exp(-30.0*s) +
            12.4*exp(-6.0*s) -
            12.0*exp(-2.0*s) +
            1.0*(0.5 + 7.0*I)*exp(-1.0*s)
        )
    )
