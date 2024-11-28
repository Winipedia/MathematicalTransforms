from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.z import ZTransform


class TestZTransform(BaseTestTransform):
    _transform_class = ZTransform

    _transform_function_to_solution_dict = {
        # ImplementationLeftAsExerciseForTheUserError
    }

    values_str = "values"
    n_values_str = "n_values"
    z_values_str = "z_values"

    _transform_data_kwargs_to_solution = (
        # Unit Step Sequence
        (
            {
                values_str: [1, 1, 1, 1],
                n_values_str: [0, 1, 2, 3],
                z_values_str: [2, 3],
            },
            [1.875, 1.481481481481]
        ),
    )
