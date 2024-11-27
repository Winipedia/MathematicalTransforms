from typing import Union, Tuple, List

import numpy as np
import sympy as sp
from sympy import abc

from transforms.base_transform.base_transform import BaseTransform


class FourierTransform(BaseTransform):

    def __init__(
            self,
            function: sp.Expr,
            is_base_form: bool = True,
            t: sp.Symbol = abc.t,
            omega: sp.Symbol = abc.omega,
    ):
        self.t = t
        self.omega = omega
        super().__init__(
            function=function,
            is_base_form=is_base_form
        )

    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic Fourier transform of the base function.
        This uses sympy's fourier_transform for symbolic computation.
        """
        # Compute Fourier BaseTransform using sympy
        transform = sp.fourier_transform(self.base_func_as_func, x=self.t, k=self.omega)
        return transform

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic inverse Fourier transform of the transformed function.
        This uses sympy's inverse_fourier_transform for symbolic computation.
        """
        # Compute Inverse Fourier BaseTransform using sympy
        inverse_transform = sp.inverse_fourier_transform(self.transformed_func_as_func, k=self.omega, x=self.t)
        return inverse_transform

    @classmethod
    def transform_data(cls, values: List[sp.Number]) -> List:
        transformed_data = list(np.fft.fft(values))
        return transformed_data

    @classmethod
    def inverse_transform_data(cls, transformed_data: List[sp.Number]) -> List[sp.Number]:
        inverse_transformed_data = list(np.fft.ifft(transformed_data))
        return inverse_transformed_data
