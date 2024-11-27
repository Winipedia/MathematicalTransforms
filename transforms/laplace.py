from typing import List, Tuple, Union

import sympy as sp
from sympy import abc

from transforms.base_transform.base_transform import BaseTransform


class LaplaceTransform(BaseTransform):

    def __init__(
            self,
            function: sp.Expr,
            is_base_form: bool = True,
            s: sp.Symbol = abc.s,
            t: sp.Symbol = abc.t,
    ):
        self.s = s
        self.t = t
        super().__init__(
            function=function,
            is_base_form=is_base_form,
        )

    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        transform = sp.laplace_transform(self.base_func_as_func, s=self.s, t=self.t)
        return transform

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        inverse_transform = sp.inverse_laplace_transform(self.transformed_func_as_func, s=self.s, t=self.t)
        return inverse_transform

    @classmethod
    def transform_data(
            cls,
            values: List[sp.Number],
            time_points: List[sp.Number],
            s_values: List[sp.Number]
    ) -> List[sp.Number]:
        """
        Compute the Laplace BaseTransform for a discrete list of points.

        Parameters:
        - values: list of function values (e.g., [1, 2, 3, 4]).
        - time_points: list of time points corresponding to the values (e.g., [0, 1, 2, 3]).
        - s_values: list of s-values for which the Laplace BaseTransform is computed.

        Returns:
        - A list of Laplace BaseTransform results for the given s-values (numerical).
        """
        if len(values) != len(time_points):
            raise ValueError("The lengths of 'values' and 'time_points' must be equal.")

        results = []
        for s in s_values:
            laplace_result = sum(v * sp.exp(-s * t).evalf() for v, t in zip(values, time_points))
            results.append(laplace_result)

        return results

    @classmethod
    def inverse_transform_data(cls, *_, **__):
        """
        The inverse of the computation above can only be approximated because of the loss of information
        as the laplace transform is made for continuous functions on the complex plane and nor for discrete
        points like it is possible for Fourier.
        The above implementation is just one that makes sense if you would attempt to transform discrete points
        and treats the continuous Integral as a summation.
        Reverting the above process is only possible in specific cases or computationally to expensive.
        """
        raise RuntimeError("Exact inversion for discrete Laplace BaseTransform is computationally infeasible or undefined.")

