from typing import Union, Tuple, List

import sympy as sp
from mpmath import mpc
from sympy import abc

from transforms.base_transform.base_transform import Transform


class FourierTransform(Transform):

    def _compute_transform_function(self, t=abc.t, omega=abc.omega) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic Fourier transform of the base function.
        This uses sympy's fourier_transform for symbolic computation.
        """
        # Compute Fourier Transform using sympy
        transform = sp.fourier_transform(self.base_function, x=t, k=omega)
        if isinstance(transform, sp.FourierTransform):
            transform = transform.as_integral.simplify()
        return transform

    def _compute_inverse_transform_function(self, t=abc.t, omega=abc.omega) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic inverse Fourier transform of the transformed function.
        This uses sympy's inverse_fourier_transform for symbolic computation.
        """
        # Compute Inverse Fourier Transform using sympy
        inverse_transform = sp.inverse_fourier_transform(self.transformed_function, k=omega, x=t)
        return inverse_transform

    @classmethod
    def transform_data(cls, values: List[mpc]) -> Union[sp.Expr]:
        transformed_data = sp.fft(values)
        return transformed_data

    @classmethod
    def inverse_transform_data(cls, transformed_data: List[mpc]) -> Union[List[mpc]]:
        inverse_transformed_data = sp.ifft(transformed_data)
        return inverse_transformed_data
