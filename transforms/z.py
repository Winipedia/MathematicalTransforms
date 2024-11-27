from typing import Union, Tuple, List

import sympy as sp

from transforms.base_transform.base_transform import BaseTransform


class ZTransform(BaseTransform):

    def _compute_transform_function(self) -> Union[Tuple, sp.Expr]:
        pass

    def _compute_inverse_transform_function(self) -> Union[Tuple, sp.Expr]:
        pass

    @classmethod
    def transform_data(cls, *args, **kwargs) -> List[sp.Number]:
        pass

    @classmethod
    def inverse_transform_data(cls, *args, **kwargs) -> List[sp.Number]:
        pass
