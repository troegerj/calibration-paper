# Standard library imports
import unittest

# Third-party imports
import jax.numpy as jnp
from jax.tree_util import tree_structure

# Local library imports
from calibrationpinn.assertions import (
    assert_equal,
    assert_equal_arrays,
    assert_equal_pytree_definition,
    assert_equal_pytrees,
)
from calibrationpinn.utilities import (
    sum_pytrees,
    mul_pytrees,
    ParametersDefinition,
    parameters_to_array,
    array_to_parameters,
)


class TestSumPytrees(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self._sut = sum_pytrees
        self._pytree_1 = {
            "model_0": {
                "parameters_0": 1.0,
                "parameters_1": jnp.array([1.0]),
                "parameters_2": jnp.array([1.0, 1.0]),
                "parameters_3": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([1.0]), jnp.array([1.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([1.0]),
                    "subparameters_1": jnp.array([1.0]),
                },
            },
        }
        self._pytree_2 = {
            "model_0": {
                "parameters_0": 2.0,
                "parameters_1": jnp.array([2.0]),
                "parameters_2": jnp.array([2.0, 2.0]),
                "parameters_3": jnp.array([[2.0, 2.0], [2.0, 2.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([2.0]), jnp.array([2.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([2.0]),
                    "subparameters_1": jnp.array([2.0]),
                },
            },
        }
        self._pytree_3 = {
            "model_0": {
                "parameters_0": 3.0,
                "parameters_1": jnp.array([3.0]),
                "parameters_2": jnp.array([3.0, 3.0]),
                "parameters_3": jnp.array([[3.0, 3.0], [3.0, 3.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([3.0]), jnp.array([3.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([3.0]),
                    "subparameters_1": jnp.array([3.0]),
                },
            },
        }

    def test_sum_two_pytrees(self) -> None:
        """
        Test that the sum of a list of two pytrees is calculated correctly.
        """
        actual = self._sut([self._pytree_1, self._pytree_2])

        expected = {
            "model_0": {
                "parameters_0": 3.0,
                "parameters_1": jnp.array([3.0]),
                "parameters_2": jnp.array([3.0, 3.0]),
                "parameters_3": jnp.array([[3.0, 3.0], [3.0, 3.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([3.0]), jnp.array([3.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([3.0]),
                    "subparameters_1": jnp.array([3.0]),
                },
            },
        }
        assert_equal_pytrees(self=self, expected=expected, actual=actual)

    def test_sum_three_pytrees(self) -> None:
        """
        Test that the sum of a list of two gradients is calculated correctly.
        """
        actual = self._sut([self._pytree_1, self._pytree_2, self._pytree_3])

        expected = {
            "model_0": {
                "parameters_0": 6.0,
                "parameters_1": jnp.array([6.0]),
                "parameters_2": jnp.array([6.0, 6.0]),
                "parameters_3": jnp.array([[6.0, 6.0], [6.0, 6.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([6.0]), jnp.array([6.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([6.0]),
                    "subparameters_1": jnp.array([6.0]),
                },
            },
        }
        assert_equal_pytrees(self=self, expected=expected, actual=actual)


class TestMulPytrees(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self._pytree = {
            "model_0": {
                "parameters_0": 1.0,
                "parameters_1": jnp.array([1.0]),
                "parameters_2": jnp.array([1.0, 1.0]),
                "parameters_3": jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([1.0]), jnp.array([1.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([1.0]),
                    "subparameters_1": jnp.array([1.0]),
                },
            },
        }
        self._factor = 2.0

    def setUp(self) -> None:
        self._sut = mul_pytrees

    def test_mul_pytrees(self) -> None:
        """
        Test that the result from multiplying a pytree with a factor is calculated correctly.
        """
        actual = self._sut(self._factor, self._pytree)

        expected = {
            "model_0": {
                "parameters_0": 2.0,
                "parameters_1": jnp.array([2.0]),
                "parameters_2": jnp.array([2.0, 2.0]),
                "parameters_3": jnp.array([[2.0, 2.0], [2.0, 2.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([2.0]), jnp.array([2.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([2.0]),
                    "subparameters_1": jnp.array([2.0]),
                },
            },
        }
        assert_equal_pytrees(self=self, expected=expected, actual=actual)


class TestParametersToArray(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.parameters_pytree = {
            "model_0": {
                "parameters_0": jnp.float32(0.0),
                "parameters_1": jnp.float64(1.0),
                "parameters_2": jnp.array([2.0]),
                "parameters_3": jnp.array([3.0, 4.0]),
                "parameters_4": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([9.0]), jnp.array([10.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([11.0]),
                    "subparameters_1": jnp.array([12.0]),
                },
            },
        }
        self._sut = parameters_to_array

    def test_parameters(self) -> None:
        """
        Test that the parameters pytree is correctly converted into an array.
        """
        actual, _ = self._sut(self.parameters_pytree)

        expected = jnp.array(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_parameters_definition_tree_definition(self):
        """
        Test that the tree definition is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.tree_definition

        expected = tree_structure(self.parameters_pytree)
        assert_equal_pytree_definition(self=self, expected=expected, actual=actual)

    def test_parameters_definition_shapes(self):
        """
        Test that the list of shapes is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.shapes

        expected = [(), (), (1,), (2,), (2, 2), (1,), (1,), (1,), (1,)]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_parameters_definition_sizes(self):
        """
        Test that the list of sizes is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.sizes

        expected = [1, 1, 1, 2, 4, 1, 1, 1, 1]
        assert_equal(self=self, expected=expected, actual=actual)


class TestParametersToArrayForOnlyOneArray(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.parameters_pytree = jnp.array([0.0, 1.0])
        self._sut = parameters_to_array

    def test_parameters(self) -> None:
        """
        Test that the parameters pytree is correctly converted into an array.
        """
        actual, _ = self._sut(self.parameters_pytree)

        expected = jnp.array([0.0, 1.0])
        assert_equal_arrays(self=self, expected=expected, actual=actual)

    def test_parameters_definition_tree_definition(self):
        """
        Test that the tree definition is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.tree_definition

        expected = tree_structure(self.parameters_pytree)
        assert_equal_pytree_definition(self=self, expected=expected, actual=actual)

    def test_parameters_definition_shapes(self):
        """
        Test that the list of shapes is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.shapes

        expected = [(2,)]
        assert_equal(self=self, expected=expected, actual=actual)

    def test_parameters_definition_sizes(self):
        """
        Test that the list of sizes is correct.
        """
        _, parameters_definition = self._sut(self.parameters_pytree)
        actual = parameters_definition.sizes

        expected = [2]
        assert_equal(self=self, expected=expected, actual=actual)


class TestArrayToParameters(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.parameters_pytree = {
            "model_0": {
                "parameters_0": jnp.float32(0.0),
                "parameters_1": jnp.float64(1.0),
                "parameters_2": jnp.array([2.0]),
                "parameters_3": jnp.array([3.0, 4.0]),
                "parameters_4": jnp.array([[5.0, 6.0], [7.0, 8.0]]),
            },
            "model_1": {
                "parameters_0": (jnp.array([9.0]), jnp.array([10.0])),
                "parameters_1": {
                    "subparameters_0": jnp.array([11.0]),
                    "subparameters_1": jnp.array([12.0]),
                },
            },
        }
        self._parameters_array = jnp.array(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        tree_definition = tree_structure(self.parameters_pytree)
        shapes = [(), (), (1,), (2,), (2, 2), (1,), (1,), (1,), (1,)]
        sizes = [1, 1, 1, 2, 4, 1, 1, 1, 1]
        self._parameters_definition = ParametersDefinition(
            tree_definition, shapes, sizes
        )
        self._sut = array_to_parameters

    def test_parameter_pytree(self) -> None:
        """
        Test that the parameters array is correctly converted into a pytree.
        """
        actual = self._sut(self._parameters_array, self._parameters_definition)

        expected = self.parameters_pytree
        assert_equal_pytrees(self=self, expected=expected, actual=actual)


class TestArrayToParametersForOnlyOneArray(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)

    def setUp(self) -> None:
        self.parameters_pytree = jnp.array([0.0, 1.0])
        self._parameters_array = jnp.array([0.0, 1.0])
        tree_definition = tree_structure(self.parameters_pytree)
        shapes = [(2,)]
        sizes = [2]
        self._parameters_definition = ParametersDefinition(
            tree_definition, shapes, sizes
        )
        self._sut = array_to_parameters

    def test_parameter_pytree(self) -> None:
        """
        Test that the parameters array is correctly converted into a pytree.
        """
        actual = self._sut(self._parameters_array, self._parameters_definition)

        expected = self.parameters_pytree
        assert_equal_pytrees(self=self, expected=expected, actual=actual)
