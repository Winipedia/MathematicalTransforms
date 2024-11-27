from typing import Union, Tuple, List
import sympy as sp
from sympy import abc
from sympy.integrals.transforms import hankel_transform, inverse_hankel_transform
from transforms.base_transform.base_transform import BaseTransform


class HankelTransform(BaseTransform):
    def __init__(
        self,
        function: sp.Expr,
        order: Union[int, float] = 0,
        is_base_form: bool = True,
        r: sp.Symbol = abc.r,
        k: sp.Symbol = abc.k,
    ):
        """
        Initialize the Hankel BaseTransform.
        :param function: The function to transform.
        :param order: The order of the Bessel function used in the transform.
        :param is_base_form: Whether the provided function is in base form or transformed form.
        :param r: The radial variable.
        :param k: The frequency variable.
        """
        self.r = r
        self.k = k
        self.order = order  # ν: Order of the Bessel function
        super().__init__(
            function=function,
            is_base_form=is_base_form,
        )

    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic Hankel transform using SymPy's built-in function.
        """
        return hankel_transform(self.base_func_as_func, r=self.r, k=self.k, nu=self.order)

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        """
        Compute the symbolic inverse Hankel transform using SymPy's built-in function.
        """
        return inverse_hankel_transform(self.transformed_func_as_func, k=self.k, r=self.r, nu=self.order)

    @classmethod
    def transform_data(
            cls,
            values: List[sp.Number],
            r_vals: List[sp.Number],
            k_vals: List[sp.Number],
            order: int
    ) -> List[sp.Number]:
        """
        Compute the Discrete Hankel BaseTransform for a discrete list of points.

        Parameters:
        - values: List of function values (e.g., [1, 2, 3, 4]).
        - r_vals: List of radial distance values corresponding to the function values.
        - k_vals: List of k-values for which the Hankel BaseTransform is computed.
        - order: Order of the Bessel function (ν).

        Returns:
        - A list of Hankel BaseTransform results for the given k-values (numerical).
        """
        if len(values) != len(r_vals):
            raise ValueError("The lengths of 'values' and 'r_vals' must be equal.")

        # Compute the Hankel BaseTransform for each k-value
        results = []
        for k in k_vals:
            hankel_result = sum(
                v * sp.besselj(order, k * r).evalf() * r
                for v, r in zip(values, r_vals)
            )
            results.append(hankel_result)

        return results

    @classmethod
    def inverse_transform_data(cls, *_, **__):
        """
        The inverse of the computation above can only be approximated because of the loss of information
        during the forward Hankel BaseTransform. The Hankel BaseTransform is inherently designed for continuous
        radially symmetric functions, and transforming discrete points is an approximation of the continuous process.

        Reverting the discrete Hankel BaseTransform is either computationally expensive or infeasible in general cases,
        as the numerical process does not perfectly map back to the original signal due to potential sampling and kernel
        inaccuracies.
        """
        raise RuntimeError(
            "Exact inversion for the Discrete Hankel BaseTransform is computationally infeasible or undefined in general cases."
        )