import numpy as np
from matplotlib import pyplot as plt

from testing.base_tests.base_transform_test import BaseTestTransform
from transforms.radon import RadonTransform


class TestRadonTransform(BaseTestTransform):

    _transform_class = RadonTransform

    _transform_function_to_solution_dict = {
        # no tests due to sympy's difficulties to simplify integrals properly
    }

    _transform_data_kwargs_to_solution = (
        # can not really be tested well, but on ask can be demonstrated during the presentation
    )

    def demonstrate_radon(self):
        # Example usage
        # Create a sample 2D array (e.g., an image)
        data = np.zeros((100, 100))
        data[40:60, 40:60] = 1  # Add a square of intensity

        # Perform Radon Transform
        sinogram = self._transform_class.transform_data(data)

        # Perform Inverse Radon Transform
        reconstructed_data = self._transform_class.inverse_transform_data(sinogram)

        # Display the results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(data, cmap='gray')
        axes[0].set_title("Original Data")
        axes[1].imshow(sinogram, cmap='gray', aspect='auto')
        axes[1].set_title("Radon Transform (Sinogram)")
        axes[2].imshow(reconstructed_data, cmap='gray')
        axes[2].set_title("Reconstructed Data")
        plt.tight_layout()
        plt.show()
