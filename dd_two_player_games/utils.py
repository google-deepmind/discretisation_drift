# Copyright 2021 DeepMind Technologies Limited and Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities."""

import numbers
import jax
import jax.numpy as jnp
import numpy as np


def reduce_sum_tree(tree):
  tree = jax.tree_util.tree_map(jnp.sum, tree)
  return jax.tree_util.tree_reduce(jnp.add, tree)


def tree_square_norm(tree):
  return reduce_sum_tree(jax.tree_map(jnp.square, tree))


def tree_mul(tree, constant):
  return jax.tree_map(lambda x: x * constant, tree)


def tree_add(tree1, tree2):
  return jax.tree_multimap(lambda x, y: x + y, tree1, tree2)


def tree_diff(tree1, tree2):
  return jax.tree_multimap(lambda x, y: x - y, tree1, tree2)


def tree_dot_product(tree1, tree2):
  prods = jax.tree_multimap(lambda x, y: x * y, tree1, tree2)
  return reduce_sum_tree(prods)


def param_size(tree):
  return jax.tree_util.tree_reduce(
      jnp.add, jax.tree_map(lambda x: x.size, tree))


def tree_shape(tree):
  return jax.tree_map(lambda x: jnp.array(x.shape), tree)


def add_trees_with_coeff(*, acc, mul, coeff):
  if coeff != 0:
    acc = tree_add(acc, tree_mul(mul, coeff))
  return acc


def batch_l2_norms(x, eps=1e-5, start_reduction=1):
  reduction_axis = list(range(start_reduction, x.ndim))
  squares = jnp.sum(jnp.square(x), axis=reduction_axis)
  return jnp.sqrt(eps + squares)


def assert_tree_equal(actual, desired):
  def assert_fn(x, y):
    equal = jnp.array_equal(x, y)
    if not equal:
      print('pp_print')
      print(x)
      print(y)
    return equal

  jax.tree_multimap(assert_fn, actual, desired)


def any_non_zero(iterable_arg):
  acc = 0
  for arg in iterable_arg:
    if not isinstance(arg, numbers.Number):
      raise ValueError('Non numeric argument in any_non_zero!')
    acc = acc or (arg != 0)
  return int(acc)


def average_dict_values(metrics):
  """Average *all* values in all .

  Args:
    metrics: A dictionary from key to tree.

  Returns:
    A dictionary from key to tree, where each value of the tree has been
    replaced with its mean across all axes.
  """
  np_avg = lambda t: np.array(t).mean()
  output = {}
  for k, value in metrics.items():
    output[k] = jax.tree_util.tree_map(np_avg, value)
  return output
