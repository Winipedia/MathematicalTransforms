from sympy import Expr

from tests.base_test.base_transform_test import TestTransform
from transforms.wavelet import WaveletTransform


class TestWaveletTransform(TestTransform):

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
