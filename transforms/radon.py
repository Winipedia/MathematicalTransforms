from typing import Union, Tuple

import numpy as np
import sympy as sp
from skimage.transform import radon, iradon
from sympy import abc

from transforms.base_transform.base_transform import BaseTransform


class RadonTransform(BaseTransform):

    def __init__(
            self,
            function: sp.Expr,
            is_base_form: bool = True,
            t: sp.Symbol = abc.t,
            theta: sp.Symbol = abc.theta,
            x: sp.Symbol = abc.x,
            y: sp.Symbol = abc.y
    ):
        self.t = t
        self.theta = theta
        self.x = x
        self.y = y
        super().__init__(
            function=function,
            is_base_form=is_base_form
        )

    def _compute_transform_function(self) -> Union[Tuple, sp.Expr]:
        transform = self.radon_transform(
            f=self.base_func_as_func,
            t=self.t,
            theta=self.theta,
        )
        return transform

    def _compute_inverse_transform_function(self, *args, **kwargs) -> Union[Tuple, sp.Expr]:
        inverse_transform = self.inverse_radon_transform(
            g=self.transformed_func_as_func,
            x=self.x,
            y=self.y,
        )
        return inverse_transform

    @classmethod
    def radon_transform(cls, f, t, theta):
        """
        Compute the symbolic Radon Transform of a 2D function.

        Parameters:
        - f: The function f(x, y) to transform.
        - t: Symbol representing the Radon Transform parameter (distance from origin).
        - theta: Symbol representing the angle of the line projection.

        Returns:
        - The symbolic Radon Transform as a function of t and theta.
        """

        x, y = abc.x, abc.y

        # Define the Dirac delta function for the line equation
        delta = sp.DiracDelta(t - x * sp.cos(theta) - y * sp.sin(theta))

        # Perform the integration
        radon = sp.integrate(f * delta, (x, -sp.oo, sp.oo), (y, -sp.oo, sp.oo))
        return radon

    @classmethod
    def inverse_radon_transform(cls, g, x, y):
        """
        Compute the symbolic inverse Radon Transform.

        Parameters:
        - g: The function g(t, theta), which is the Radon Transform of a 2D function.
        - x: Symbol representing the x-coordinate in the spatial domain.
        - y: Symbol representing the y-coordinate in the spatial domain.

        Returns:
        - The symbolic inverse Radon Transform as a function of x and y.
        """

        # Define the symbols
        t, theta = sp.symbols('t theta', real=True)

        # Define the Dirac delta function for the line equation
        delta = sp.DiracDelta(t - x * sp.cos(theta) - y * sp.sin(theta))

        # Add normalization factor (1/2π)
        normalization_factor = 1 / (2 * sp.pi)

        # Perform the double integration
        inverse_radon = normalization_factor * sp.integrate(g * delta, (t, -sp.oo, sp.oo), (theta, 0, sp.pi))

        return inverse_radon

    @classmethod
    def transform_data(
            cls,
            data: np.ndarray,
            angles: np.ndarray = None,
            circle: bool = True
    ) -> np.ndarray:
        """
        Apply Radon Transform to the given 2D data.

        Parameters:
            data (np.ndarray): 2D array (e.g., an image or grid) to transform.
            angles (np.ndarray, optional): Array of angles (in degrees) for the transform.
                                           Defaults to 0 to 180 degrees evenly spaced.
            circle (bool, optional): If True, assume the input data is circular.
                                     If False, the input is treated as rectangular.

        Returns:
            np.ndarray: The Radon transform (sinogram) of the input data.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array.")

        if angles is None:
            # Default to evenly spaced angles from 0 to 180 degrees
            angles = np.linspace(0., 180., max(data.shape), endpoint=False)

        return radon(data, theta=angles, circle=circle)

    @classmethod
    def inverse_transform_data(
            cls,
            sino_gram: np.ndarray,
            angles: np.ndarray = None,
            circle: bool = True
    ) -> np.ndarray:
        """
        Perform the Inverse Radon Transform to reconstruct 2D data from a sinogram.

        Parameters:
            sino_gram (np.ndarray): The Radon Transform (sinogram) as a 2D array.
            angles (np.ndarray, optional): Array of angles (in degrees) corresponding to the sinogram.
                                           Defaults to 0 to 180 degrees evenly spaced.
            circle (bool, optional): If True, assumes the input data is circular. Defaults to True.

        Returns:
            np.ndarray: The reconstructed 2D array (e.g., an image).
        """
        if not isinstance(sino_gram, np.ndarray) or sino_gram.ndim != 2:
            raise ValueError("Input sinogram must be a 2D NumPy array.")

        if angles is None:
            # Default to evenly spaced angles from 0 to 180 degrees
            angles = np.linspace(0., 180., sino_gram.shape[1], endpoint=False)

        # Perform the inverse Radon Transform
        return iradon(sino_gram, theta=angles, circle=circle)
