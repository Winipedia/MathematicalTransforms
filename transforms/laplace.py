from typing import List, Tuple, Union

import sympy as sp
from mpmath import mpc
from sympy import abc

from transforms.base_transform.base_transform import Transform
from utils.consts import ZERO
from utils.mpc import generate_standard_values_for_axis, generate_deltas_for_summation, to_mpc, \
    subs_zero


class LaplaceTransform(Transform):

    def _compute_transform_function(self, s=abc.s, t=abc.t) -> Union[Tuple, sp.Basic]:
        transform = sp.laplace_transform(self.base_function, s=s, t=t)
        return transform

    def _compute_inverse_transform_function(self, s=abc.s, t=abc.t) -> Union[Tuple, sp.Basic]:
        inverse_transform = sp.inverse_laplace_transform(self.transformed_function, s=s, t=t)
        return inverse_transform

    @classmethod
    def transform_data(
            cls,
            f_values: List[mpc],
            t_values: List[mpc] = None
    ) -> sp.Expr:
        """
        This function allows a laplace transform on discrete data-points
        :param f_values: The values of f(t_i)
        :param t_values: The time points the f_values correspond to
        :return: the transformed laplace sum
        """

        # t can be None and will receive standard 1,2,3,4... values
        if t_values is None:
            t_values = generate_standard_values_for_axis(f_values)

        # calculate delta ts
        delta_ts = generate_deltas_for_summation(t_values)

        # assert all t_values are increasing from one to the next
        assert len(f_values) == len(t_values) == len(delta_ts), \
            f"{len(f_values)=} != {len(t_values)=} != {len(delta_ts)}"

        # Loop over each data point to compute the laplace sum
        laplace_sum_expr = mpc(0, 0)
        for f_i, t_i, delta_t in zip(f_values, t_values, delta_ts):
            # Compute the term for each data point
            if f_i == 0:
                # so that 0 terms in f_i won't get lost bc they'd automatically simplify
                # and could not be backtracked when inverting
                f_i = ZERO
            term = f_i * sp.exp(-abc.s * t_i) * delta_t
            laplace_sum_expr += term

        return laplace_sum_expr

    @classmethod
    def inverse_transform_data(
            cls,
            laplace_sum_expr: sp.Expr,
    ) -> Tuple[List[mpc], List[mpc]]:
        """
        This function is used to inverse the action from transform_data() func
        :param laplace_sum_expr: The sum Expression you got from transforming a data array
        :return: the original data points and t_values
        """
        # Expand the expression to get individual terms
        terms = laplace_sum_expr.as_ordered_terms()

        t_val_to_f_val_times_delta_t = {}
        for term in terms:
            # Extract the exponential part
            exp_terms = term.atoms(sp.exp)
            if not exp_terms:
                # means t_i must have been 0
                t_i = mpc(0, 0)
                coefficient = term
            else:
                exp_term = list(exp_terms)[0]
                exponent = exp_term.args[0]  # Exponent should be -s * t_i

                # Solve for t_i
                t_i_expr = -exponent / abc.s
                t_i = t_i_expr.evalf()

                # Extract the coefficient (f_i * delta_t)
                coefficient = term / exp_term
                coefficient = sp.N(coefficient)

            t_i = to_mpc(t_i)
            if isinstance(coefficient, sp.Expr):
                coefficient = subs_zero(coefficient)
            coefficient = to_mpc(coefficient)
            t_val_to_f_val_times_delta_t[t_i] = coefficient

        # order the t val by the real part
        t_val_to_f_val_times_delta_t = dict(sorted(t_val_to_f_val_times_delta_t.items(), key=lambda items: items[0].real))

        t_values = list(t_val_to_f_val_times_delta_t.keys())
        # Reconstruct delta_ts from t_values
        delta_ts = generate_deltas_for_summation(t_values)

        # Calculate f_values
        f_values = []
        for f_val_dt, delta_t in zip(t_val_to_f_val_times_delta_t.values(), delta_ts):
            f_i = f_val_dt / delta_t
            f_values.append(f_i)

        return f_values, t_values
