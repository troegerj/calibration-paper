# Standard library imports
import numbers
from typing import Any, Callable, Union, Type
from unittest import TestCase


# Third-party imports
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import pandas as pd

# Local library imports
from calibrationpinn.testConfig import TestConfig
from calibrationpinn.typeAliases import (
    JNPPyTreeDef,
    JNPPyTree,
    Array,
    JNPArray,
    PDDataFrame,
    PYNumbers,
    PRNGKey,
)


def assert_equal(self: TestCase, expected: Any, actual: Any) -> None:
    self.assertEqual(expected, actual)


def assert_almost_equal(self: TestCase, expected: PYNumbers, actual: PYNumbers) -> None:
    self.assertAlmostEqual(expected, actual, places=TestConfig.TOLERANCE_DECIMAL_PLACE)


def assert_true(self: TestCase, expression: bool) -> None:
    self.assertTrue(expression)


def assert_false(self: TestCase, expression: bool) -> None:
    self.assertFalse(expression)


def assert_raises_error(
    self: TestCase,
    expected_exception: Type[Exception],
    func: Callable[[Any], Any],
    *func_args: Any
) -> None:
    self.assertRaises(expected_exception, func, *func_args)


# jax and numpy
def assert_equal_arrays(
    self: TestCase, expected: Array, actual: Array, atol=TestConfig.TOLERANCE_ABSOLUTE
) -> None:
    self.assertEqual(expected.shape, actual.shape)
    self.assertTrue(jnp.allclose(expected, actual, atol=atol))


def assert_equal_lists_of_arrays(
    self: TestCase, expected: list[Array], actual: list[Array]
) -> None:
    for index, array in enumerate(actual):
        assert_equal_arrays(self, expected=expected[index], actual=array)


# jax
def assert_equal_PRNGKeys(self: TestCase, expected: PRNGKey, actual: PRNGKey) -> None:
    self.assertEqual(expected.shape, actual.shape)
    self.assertTrue(jnp.allclose(expected, actual, atol=TestConfig.TOLERANCE_ABSOLUTE))


def assert_equal_pytrees(
    self: TestCase, expected: JNPPyTree, actual: JNPPyTree
) -> None:
    pytree_flat_expected, pytree_def_expected = tree_flatten(expected)
    pytree_flat_actual, pytree_def_actual = tree_flatten(actual)
    assert_equal_pytree_leaves(self, pytree_flat_expected, pytree_flat_actual)
    assert_equal_pytree_definition(self, pytree_def_expected, pytree_def_actual)


def assert_equal_pytree_leaves(self: TestCase, expected: Any, actual: Any) -> None:
    for index, actual_leave in enumerate(actual):
        if isinstance(actual_leave, jnp.ndarray):
            assert_equal_arrays(self, expected[index], actual_leave)
        else:
            assert_equal(self, expected[index], actual_leave)


def assert_equal_pytree_definition(
    self: TestCase, expected: JNPPyTreeDef, actual: JNPPyTreeDef
) -> None:
    assert_equal(self, expected, actual)


# Pandas
def assert_equal_data_frames(
    self: TestCase, expected: PDDataFrame, actual: PDDataFrame
) -> None:
    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=True,
        check_names=True,
        check_exact=False,
        atol=TestConfig.TOLERANCE_ABSOLUTE,
    )


def assert_equal_data_frames_without_index(
    self: TestCase, expected: PDDataFrame, actual: PDDataFrame
) -> None:
    assert_equal_data_frames(
        self, expected.reset_index(drop=True), actual.reset_index(drop=True)
    )
