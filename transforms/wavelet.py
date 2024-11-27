from typing import Union, Tuple, List

import sympy as sp
from mpmath import mpc

from exceptions import raise_left_as_exercise_for_reader
from transforms.base_transform.base_transform import BaseTransform


class WaveletTransform(BaseTransform):

    @raise_left_as_exercise_for_reader
    def _compute_transform_function(self) -> Union[Tuple, sp.Basic]:
        pass

    @raise_left_as_exercise_for_reader
    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Basic]:
        pass

    @classmethod
    @raise_left_as_exercise_for_reader
    def transform_data(cls, *args, **kwargs) -> Union[sp.Expr]:
        pass

    @classmethod
    @raise_left_as_exercise_for_reader
    def inverse_transform_data(cls, *args, **kwargs) -> Union[List[mpc]]:
        pass
