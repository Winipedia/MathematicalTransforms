"""
The Base class for BaseTransform objects
"""
from typing import Tuple, Union, List
import sympy as sp
from sympy.integrals.transforms import IntegralTransform

from utils.sympy_math import (
    replace_unevaled_integrals_with_forms, to_number, INF
)


class BaseTransform:

    """
    Base class for BaseTransform objects
    After initialization you got the following attributes:
        self.base_function: which is the base e.g. f(t) in Laplace Transforms
        self.transformed_function: which is the transform of the base function e.g. F(s) in Laplace Transforms

    In addition, you have the methods:

    transform_data() and inverse_transform_data() to transform data points with the transform

    """

    def __init__(
            self,
            function: sp.Expr,
            is_base_form: bool = True
    ):

        if type(self) is BaseTransform:
            raise TypeError(
                f"{self.__class__.__name__} should not be instantiated directly. "
                f"It is a base class and should be subclassed."
            )

        # validate the given function
        function = self._validate_input(function)

        # calculate the transform and inverse transform
        # the self. that is assigned to function must be referred
        # first to be able to be used in the transform or inverse transform
        if is_base_form:
            self.base_function = function
            self.transformed_function = self._transform_function()
        else:
            self.transformed_function = function
            self.base_function = self._inverse_transform_function()

    @staticmethod
    def _validate_input(function: sp.Expr | Tuple):
        if not isinstance(function, sp.Basic):
            function = to_number(function)
        if not isinstance(function, (sp.Expr, Tuple)):
            raise ValueError("function must be a sympy expression or tuple with convergence and domain")

        return function

    def _get_transform_result(self, transform_res):
        """
        Replaces all unevaluated IntegralTransform objects in a SymPy expression
        with their integral representations.

        Parameters:
            transform_res: The expression to process, which can contain IntegralTransform
                           objects or nested SymPy structures.

        Returns:
            A new expression with all unevaluated IntegralTransforms replaced by their
            integral forms.
        """
        if isinstance(transform_res, tuple):
            func = self._get_transform_result(transform_res[0])
            transform_res = (func, *transform_res[1:])
            return transform_res

        # Define a replacement rule for IntegralTransform objects
        def replace_transform(expr):
            if isinstance(expr, IntegralTransform):
                expr = expr.as_integral
                expr = replace_unevaled_integrals_with_forms(expr)
            return expr

        # Apply the replacement rule to the entire expression
        transform_res = transform_res.xreplace(
            {expr: replace_transform(expr) for expr in transform_res.atoms(IntegralTransform)}
        )

        transform_res = transform_res.replace(sp.oo, INF)

        return transform_res

    @staticmethod
    def _extract_function(func_or_tuple: Tuple | sp.Expr) -> sp.Expr:
        return func_or_tuple[0] if isinstance(func_or_tuple, tuple) else func_or_tuple

    @property
    def base_func_as_func(self):
        return self._extract_function(self.base_function)

    @property
    def transformed_func_as_func(self):
        return self._extract_function(self.transformed_function)

    """
    Transformations applied to a symbolic mathematical function
    """
    def _transform_function(self) -> Tuple:
        if not hasattr(self, "base_function"):
            raise AttributeError("base_function must be an attribute of the class before applying transform")
        transform = self._compute_transform_function()
        return self._get_transform_result(transform)

    def _compute_transform_function(self) -> Union[Tuple, sp.Expr]:
        raise NotImplementedError

    """
    Inverse Transformations applied to a symbolic mathematical function
    """
    def _inverse_transform_function(self) -> Tuple:
        if not hasattr(self, "transformed_function"):
            raise AttributeError("transformed_function must be an attribute of the class before applying inverse transform")
        inverse_transform = self._compute_inverse_transform_function()
        return self._get_transform_result(inverse_transform)

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Expr]:
        raise NotImplementedError

    """
    Data Transformations, applied to given data onto the Datapoints
    """
    @classmethod
    def transform_data(cls, *args, **kwargs) -> List[sp.Number]:
        """
        Is meant to be able to take discrete data points and transform them.
        This transformation depends on the transform and its applications
        and varies between Transforms.
        """
        raise NotImplementedError

    """
    Inverse Data Transformations, applied to given data onto the Datapoints
    """
    @classmethod
    def inverse_transform_data(cls, *args, **kwargs) -> List[sp.Number]:
        """
        Is meant to inverse transform_data() and return the arguments given to it,
        while only provided with the output of it.
        """
        raise NotImplementedError
