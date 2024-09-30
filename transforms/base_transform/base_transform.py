"""
The Base class for Transform objects
"""
from typing import Tuple, Union, List
import sympy as sp
from mpmath import mpc


class Transform:

    """
    Base class for Transform objects
    After initialization you got the following attributes:
        self.base_function: which is the base e.g. f(t) in Laplace Transforms
        self.transformed_function: which is the transform of the base function e.g. F(s) in Laplace Transforms

    In addition, you have the methods:

    transform_data() and inverse_transform_data() to transform data points with the transform

    """

    def __init__(self, function: sp.Basic, is_base_form: bool = True):

        if type(self) is Transform:
            raise TypeError(
                f"{self.__class__.__name__} should not be instantiated directly. "
                f"It is a base class and should be subclassed."
            )

        # validate the given function
        self._validate_input(function)

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
    def _validate_input(function: sp.Basic):
        if not isinstance(function, sp.Basic):
            raise ValueError("function must be a sympy expression.")

    @staticmethod
    def _extract_function(function: sp.Basic or tuple) -> sp.Basic:
        if isinstance(function, tuple):
            transform, convergence_domain, condition = function
            return transform
        elif isinstance(function, sp.Basic):
            return function
        else:
            raise ValueError("function must be a sympy expression or a tuple of (transform, convergence_domain, condition).")

    """
    Transformations applied to a symbolic mathematical function
    """
    def _transform_function(self) -> sp.Basic:
        if not hasattr(self, "base_function"):
            raise AttributeError("base_function must be an attribute of the class before applying transform")
        transform = self._compute_transform_function()
        return self._extract_function(transform)

    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        raise NotImplementedError

    """
    Inverse Transformations applied to a symbolic mathematical function
    """
    def _inverse_transform_function(self) -> sp.Basic:
        if not hasattr(self, "transformed_function"):
            raise AttributeError("transformed_function must be an attribute of the class before applying inverse transform")
        inverse_transform = self._compute_inverse_transform_function()
        return self._extract_function(inverse_transform)

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        raise NotImplementedError

    """
    Data Transformations, applied to given data onto the Datapoints
    """
    @classmethod
    def transform_data(cls, *args, **kwargs) -> Union[sp.Expr]:
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
    def inverse_transform_data(cls, *args, **kwargs) -> Union[List[mpc]]:
        """
        Is meant to inverse transform_data() and return the arguments given to it,
        while only provided with the output of it.
        """
        raise NotImplementedError
