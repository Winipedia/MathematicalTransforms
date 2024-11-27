from sympy import exp, pi, Number, DiracDelta, sqrt, I
from sympy.abc import omega, t

from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.fourier import FourierTransform


class TestFourierTransform(BaseTestTransform):
    _transform_class = FourierTransform

    _transform_function_to_solution_dict = {
        DiracDelta(t): Number(1),  # Dirac delta function
        exp(-t ** 2): sqrt(pi) * exp(-pi ** 2 * omega ** 2),
        DiracDelta(t - 1): exp(-2 * I * pi * omega)
    }

    values_str = "values"
    _transform_data_kwargs_to_solution = (
        # Delta Function (Impulse Signal)
        (
            {
                values_str: [1, 0, 0, 0],
            },
            [(1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j)],  # Flat spectrum
        ),
        # Unit Step Signal
        (
            {
                values_str: [1, 1, 1, 1],
            },
            [(4 + 0j), 0j, 0j, 0j],  # Strong DC component
        ),
        # Sinusoidal Wave
        (
            {
                values_str: [0, 1, 0, -1],
            },
            [0j, -2j, 0j, 2j],  # Two spikes for sin(2πft)
        ),
        # Cosine Wave
        (
            {
                values_str: [1, 0, -1, 0],
            },
            [0j, (2 + 0j), 0j, (2 + 0j)],  # Real-valued spikes for cos(2πft)
        ),
        # Rectangular Pulse
        (
            {
                values_str: [1, 1, 0, 0],
            },
            [(2 + 0j), (1 - 1j), 0j, (1 + 1j)],  # Sinc-like pattern
        ),
        # Linear Ramp
        (
            {
                values_str: [0, 1, 2, 3],
            },
            [(6 + 0j), (-2 + 2j), (-2 + 0j), (-2 - 2j)],  # DC with decaying spectrum
        ),
        # Random Noise
        (
            {
                values_str: [0.2, -1.3, 0.7, 0.9],
            },
            [(0.5 + 0j), (-0.5 + 2.2j), (1.3 + 0j), (-0.5 - 2.2j)],  # Broadband spectrum
        ),
        # Periodic Square Wave
        (
            {
                values_str: [1, -1, 1, -1],
            },
            [0j, 0j, (4 + 0j), 0j],  # Odd harmonics
        ),
        # Sawtooth Wave
        (
            {
                values_str: [0, 1, 2, 3],
            },
            [(6 + 0j), (-2 + 2j), (-2 + 0j), (-2 - 2j)],  # Harmonics with decreasing amplitude
        ),
    )

