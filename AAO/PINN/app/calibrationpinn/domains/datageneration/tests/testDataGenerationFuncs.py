# Standard library imports
import unittest

# Third-party imports
import jax.numpy as jnp

# Local library imports
from calibrationpinn.assertions import assert_equal_arrays
from calibrationpinn.domains.datageneration import interpolate_on_grid


class TestGridDataGenerationFunc(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = interpolate_on_grid

    def test_interpolate_on_grid(self):
        """
        Test that the solution values are correctly interpolated on the grid coordinates.
        """
        coordinates_solution = jnp.array(
            [[0.0, 2.0], [2.0, 2.0], [0.0, 0.0], [2.0, 0.0]]
        )
        values_solution = jnp.array([[20.0], [20.0], [0.0], [0.0]])
        coordinates_grid_x = jnp.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        coordinates_grid_y = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        actual = self._sut(
            coordinates_solution=coordinates_solution,
            values_solution=values_solution,
            coordinates_grid_x=coordinates_grid_x,
            coordinates_grid_y=coordinates_grid_y,
        )

        expected = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 20.0],
            ]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)
