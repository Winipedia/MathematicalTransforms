from typing import List, Tuple, Union
import sympy as sp
from sympy import abc
from transforms.base_transform.base_transform import BaseTransform


class ZTransform(BaseTransform):

    def __init__(
            self,
            function: sp.Expr,
            is_base_form: bool = True,
            z: sp.Symbol = abc.z,
            n: sp.Symbol = abc.n,
    ):
        self.z = z
        self.n = n
        super().__init__(
            function=function,
            is_base_form=is_base_form,
        )

    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic Z-transform of the base function.
        """
        transform = self.z_transform(self.base_func_as_func, self.n, self.z)
        return transform

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic inverse Z-transform of the transformed function.
        """
        inverse_transform = self.inverse_z_transform(self.transformed_func_as_func, self.n, self.z)
        return inverse_transform

    @classmethod
    def z_transform(cls, f_n, n, z):
        """
        Compute the symbolic Z-transform of a function.

        Parameters:
        - f: The function f(n) to transform.
        - n: The discrete variable (time domain).
        - z: The complex variable (Z domain).

        Returns:
        - The symbolic Z-transform.
        """
        return sp.summation(f_n * z**(-n), (n, 0, sp.oo))

    @classmethod
    def inverse_z_transform(cls, f_z, n, z):
        """
        Compute the symbolic inverse Z-transform of a function.

        Parameters:
        - F: The function F(z) in the Z domain.
        - n: The discrete variable (time domain).
        - z: The complex variable (Z domain).

        Returns:
        - The symbolic inverse Z-transform.
        """
        return sp.summation(f_z * z**n / (2 * sp.pi * sp.I), (z, sp.oo, -sp.oo))

    @classmethod
    def transform_data(
            cls,
            values: List[sp.Number],
            n_values: List[int],
            z_values: List[sp.Number]
    ) -> List[sp.Number]:
        """
        Compute the numerical Z-transform for a discrete list of points over multiple z-values.

        Parameters:
        - values: List of function values (e.g., [1, 2, 3, 4]).
        - n_values: List of corresponding n-values (e.g., [0, 1, 2, 3]).
        - z_values: List of z-values at which to evaluate the Z-transform.

        Returns:
        - A list of numerical Z-transform results for each z-value.
        """
        if len(values) != len(n_values):
            raise ValueError("The lengths of 'values' and 'n_values' must be equal.")

        results = []
        for z_value in z_values:
            z_transform = sum(v * z_value ** (-n) for v, n in zip(values, n_values))
            results.append(z_transform)

        return results

    @classmethod
    def inverse_transform_data(cls, *_, **__):
        """
        The inverse computation of the Z-transform from discrete points is not feasible.
        The Z-transform operates on sequences defined over discrete indices, and its inversion
        requires analytic continuation or symbolic computation, not discrete point-based methods.

        Raises:
            RuntimeError: Always raised since exact inversion is not feasible for discrete points.
        """
        raise RuntimeError(
            "Exact inversion of the Z-transform for discrete points is computationally infeasible or undefined."
        )
