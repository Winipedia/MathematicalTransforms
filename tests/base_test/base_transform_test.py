from typing import Dict, Type, Any, Tuple
from unittest import TestCase
from transforms.base_transform.base_transform import Transform

import sympy as sp

from utils.mpc import set_precision, deep_almost_equal
from utils.mpc import functions_are_equal


class TestTransform(TestCase):

    _transform_class: Type[Transform] = NotImplemented

    _transform_function_to_solution_dict: Dict[sp.Basic, sp.Basic] = NotImplemented

    _transform_data_kwargs_to_solution: Tuple[Tuple[Dict[str, Any], Any]] = NotImplemented

    def setUp(self) -> None:

        self.transform_class = self._transform_class
        self.transform_function_to_solution_dict = self._transform_function_to_solution_dict
        self.transform_data_kwargs_to_solution = self._transform_data_kwargs_to_solution
        self.test_dps = 1000
        set_precision(self.test_dps)
        self.test_dps_tol = 10

        if any(
            item is NotImplemented for item in (
                    self.transform_class,
                    self.transform_function_to_solution_dict,
                    self.transform_data_kwargs_to_solution
            )
        ):
            raise NotImplementedError

    """
    Test the transforming of functions
    """
    def test_transform(self):
        self._test_transform_on_functions()

    def _test_transform_on_functions(self):
        for test_func, solution in self.transform_function_to_solution_dict.items():
            self._test_transform_on_function(function=test_func, solution=solution)

    def _test_transform_on_function(
            self,
            function: sp.Basic,
            solution: sp.Basic
    ):

        transform = self.transform_class(function=function, is_base_form=True)
        self.assertEqual(transform.transformed_function, solution)
        self.assertEqual(transform.base_function, function)

        inverse_transform = self.transform_class(function=solution, is_base_form=False)
        self.assertEqual(inverse_transform.base_function, function)
        self.assertEqual(inverse_transform.transformed_function, solution)

    """
    Test the transforming of discrete data
    """
    def test_transform_data(self):
        for kwargs, solution in self.transform_data_kwargs_to_solution:
            self._test_transform_data(kwargs, solution)

    def _test_transform_data(self, kwargs, solution):
        transformed_data = self.transform_class.transform_data(**kwargs)
        self._assert_transformed_data_correct(transformed_data, solution)
        
        inverse_transformed_data = self.transform_class.inverse_transform_data(transformed_data)
        self._assert_inverse_transformed_data_correct(inverse_transformed_data, kwargs)

    def _assert_transformed_data_correct(self, transformed_data, solution):
        if isinstance(transformed_data, sp.Expr):
            self.assertTrue(
                functions_are_equal(
                    transformed_data, solution,
                    dps_tol=self.test_dps_tol, num_tests=10
                )
            )
        else:
            self.assertEqual(transformed_data, solution)

    def _assert_inverse_transformed_data_correct(self, inverse_transformed_data, kwargs):

        solution = tuple(kwargs.values())
        if len(solution) == 1:
            solution = solution[0]
        self.assertTrue(deep_almost_equal(inverse_transformed_data, solution, dps_tol=self.test_dps_tol))
