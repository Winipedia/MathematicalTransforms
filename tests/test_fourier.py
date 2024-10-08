from sympy import exp, sqrt, cos, I, pi, DiracDelta, Heaviside, Number
from sympy.abc import omega, t

from tests.base_test.base_transform_test import TestTransform
from transforms.fourier import FourierTransform


class TestFourierTransform(TestTransform):

    _transform_class = FourierTransform

    _transform_function_to_solution_dict = {
        DiracDelta(t): Number(1),  # Dirac delta function
        Heaviside(t): sqrt(2 * pi) * (1 / (I * omega) + pi * DiracDelta(omega)),
        # Heaviside step function
        exp(-t): 1 / (1 + I * omega),  # Exponential decay
        cos(t): sqrt(pi / 2) * (DiracDelta(omega - 1) + DiracDelta(omega + 1))  # Cosine function
    }

    _transform_data_kwargs_to_solution = (

    )
