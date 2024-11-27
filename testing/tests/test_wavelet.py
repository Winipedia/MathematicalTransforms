from sympy import Expr

from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.wavelet import WaveletTransform


class TestWaveletTransform(BaseTestTransform):

    _transform_class = WaveletTransform

    _transform_function_to_solution_dict = {
        Expr(1): 1,
    }

    _transform_data_kwargs_to_solution = (
        (
            {
                "smth1": [1, 2, 3],
                "smth2": [1, 2, 3],
            },
            1
        ),
    )
