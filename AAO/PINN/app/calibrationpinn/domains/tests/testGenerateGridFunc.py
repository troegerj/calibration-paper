# Standard library imports
import unittest

# Third-party imports
import numpy as np

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains.generateGridFunc import generate_grid


class TestGenerateGridFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = generate_grid

    def test_grid_generation_func_coordinates_x(self) -> None:
        """
        Test that the grid x coordinates are correctly generated from the passed coordinates array.
        """
        coordinates = np.array(
            [[1.0, 2.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 0.0]]
        )

        actual, _ = self._sut(coordinates=coordinates, num_points_per_edge=3)

        expected = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_grid_generation_func_coordinates_y(self) -> None:
        """
        Test that the grid y coordinates are correctly generated from the passed coordinates array.
        """
        coordinates = np.array(
            [[1.0, 2.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 0.0]]
        )

        _, actual = self._sut(coordinates=coordinates, num_points_per_edge=3)

        expected = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert_equal_arrays(self=self, expected=expected, actual=actual)
