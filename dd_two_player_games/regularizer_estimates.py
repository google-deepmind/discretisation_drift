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

r"""Different estimators for the regularizers.

Functions in this file are used to estimate quantities of the form:
 \nabla_{\phi} (
    \nabla_{\theta} L_1(\theta, \phi) ^T
          stop_grad(\nabla_{\theta} L_2(\theta, \phi)))
  where L_1, L_2 are loss functions which contains expectations.
"""

import jax
import jax.numpy as jnp

from dd_two_player_games import utils


def _select_grad(grad_fn, grad_var):
  return lambda *args: grad_fn(*args)._asdict()[grad_var]


def biased_estimate_grad_fn(fn1, fn2, grad_var='disc'):
  r"""Function which computes an unbiased estimate of the regularizer grads.

  Estimates:
    \nabla_{\phi} (
      \nabla_{\theta} L_1(\theta, \phi) ^T \nabla_{\theta} L_2(\theta, \phi))

  This returns a biased estimate, by using the same set of samples from the
    expectations in L_1, and L_2 when computing them.

  This estimator does not return the same results if multiple devices are used.

  Args:
    fn1: A function which will return the first set of gradients used for the
      regularizers. These gradients will be backpropagated through.
      Returns \nabla_{\theta} L_1(\theta, \phi).
    fn2: A function which will return the second set of gradients used for the
      regularizers. These gradients will be NOT backpropagated through.
      Returns \nabla_{\theta} L_2(\theta, \phi).
    grad_var: A string - 'disc' or 'gen' - the variable with respect to which
      to compute the gradients.

  Returns:
    A gradient function which can be applied to compute the desired gradients.
  """
  def estimate(params, state, data_batch, rng, is_training):
    value1 = fn1(params, state, data_batch, rng, is_training)
    value2 = fn2(params, state, data_batch, rng, is_training)
    value2 = jax.lax.stop_gradient(value2)
    return utils.tree_dot_product(value1, value2)

  return _select_grad(jax.grad(estimate), grad_var)


def unbiased_estimate_fn_different_device_results(fn1, fn2, grad_var='disc'):
  return _unbiased_estimate_fn_cross_product_samples(
      fn1, fn2, False, grad_var=grad_var)


def unbiased_estimate_fn_lowest_variance(fn1, fn2, grad_var='disc'):
  return _unbiased_estimate_fn_cross_product_samples(
      fn1, fn2, True, grad_var=grad_var)


def _unbiased_estimate_fn_cross_product_samples(
    fn1, fn2, use_pmap_on_stop_grad_tree, grad_var='disc'):
  r"""Function which computes an unbiased estimate of the regularizer grads.

  Estimates:
    \nabla_{\phi} (
      \nabla_{\theta} L_1(\theta, \phi) ^T \nabla_{\theta} L_2(\theta, \phi))

  It computes:
    1/ N^2 \sum_{i=1}{N} (
      \nabla_{GRAD_VAR} (\nabla_{\theta} L_1(\theta, \phi)(x_1,i) ^T))
     \sum_{j=1}{N} \nabla_{\theta} L_1(\theta, \phi)(x_2,i) ^T

    where x_1, i, and x_2, i are iid samples from the distributions
      required to compute L_1 and L_2.

  This returns a biased estimated, by using the same set of samples from the
    expectations in L_1, and L_2 when computing them.

  This estimator does not return the same results if multiple devices are used.

  Variance of this estimator scaled with 1/ (N^2 P), where N is the batch size
  per device, and P is the number of devices.

  Args:
    fn1: A function which will return the first set of gradients used for the
      regularizers. These gradients will be backpropagated through.
      Returns \nabla_{\theta} L_1(\theta, \phi).
    fn2: A function which will return the second set of gradients used for the
      regularizers. These gradients will be NOT backpropagated through.
      Returns \nabla_{\theta} L_2(\theta, \phi).
    use_pmap_on_stop_grad_tree: A boolean deciding whether or not to use
      pmap on the tree for which we apply stop_gradient. If True, this is
      results in a lower variance estimate.
    grad_var: A string - 'disc' or 'gen' - the variable with respect to which
      to compute the gradients.

  Returns:
    A gradient function which can be applied to compute the desired gradients.
  """
  def dot_prod(tree2, params, state, x1, rng1, is_training):
    tree1 = fn1(params, state, x1, rng1, is_training)
    tree2 = jax.lax.stop_gradient(tree2)
    return utils.tree_dot_product(tree1, tree2)

  def vmap_grad(params, state, data_batch, rng, is_training):
    data1, data2 = jnp.split(data_batch, [int(data_batch.shape[0]/2),], axis=0)

    # Split rngs so that we can vmap the operations and obtained an unbiased
    # estimate. Note: this means that we will obtain a different sample
    # for each vmap, but due to the split we will also obtain a different
    # set of samples than those obtained in the forward pass.
    rng1, rng2 = jax.random.split(rng, 2)

    # The non gradient part gets computed and averaged first.
    value2 = fn2(params, state, data2, rng2, is_training)

    # We pmean over all devices. This results in a lower variance
    # estimates but keeps the unbiased property of the estimator since
    # it avoids backpropagating through a gradient operation.
    if use_pmap_on_stop_grad_tree:
      value2 = jax.lax.pmean(value2, axis_name='i')

    grad_fn = _select_grad(jax.grad(dot_prod, argnums=1), grad_var)
    return grad_fn(value2, params, state, data1, rng1, is_training)

  return vmap_grad


def biased_estimate_multiple_devices_grad_fn(fn1, fn2, grad_var='disc'):
  return biased_estimate_general_grad_fn(fn1, fn2, True, grad_var=grad_var)


def biased_estimate_general_grad_fn(
    fn1, fn2, use_pmap_on_stop_grad_tree, grad_var='disc'):
  r"""Function which computes a biased estimate of the regularizer grads.

    This biased estimate obtains the same results regardless of the number
    of devices used.

  Estimates:
    \nabla_{\phi} (
      \nabla_{\theta} L_1(\theta, \phi) ^T \nabla_{\theta} L_2(\theta, \phi))

  It computes:
    1/ N^2 \sum_{i=1}{N} (
      \nabla_{GRAD_VAR} (\nabla_{\theta} L_1(\theta, \phi)(x_1,i) ^T))
     \sum_{j=1}{N} \nabla_{\theta} L_1(\theta, \phi)(x_2,i) ^T

    where x_1, i, and x_2, i are iid samples from the distributions
      required to compute L_1 and L_2.

  This returns a biased estimated, by using the same set of samples from the
    expectations in L_1, and L_2 when computing them.

  Args:
    fn1: A function which will return the first set of gradients used for the
      regularizers. These gradients will be backpropagated through.
      Returns \nabla_{\theta} L_1(\theta, \phi).
    fn2: A function which will return the second set of gradients used for the
      regularizers. These gradients will be NOT backpropagated through.
      Returns \nabla_{\theta} L_2(\theta, \phi).
    use_pmap_on_stop_grad_tree: A boolean deciding whether or not to use
      pmap on the tree for which we apply stop_gradient. If True, this is
      results in a lower variance estimate.
    grad_var: A string - 'disc' or 'gen' - the variable with respect to which
      to compute the gradients.

  Returns:
    A gradient function which can be applied to compute the desired gradients.
  """
  def dot_prod(tree2, params, state, x, rng, is_training):
    tree1 = fn1(params, state, x, rng, is_training)
    tree2 = jax.lax.stop_gradient(tree2)
    return utils.tree_dot_product(tree1, tree2)

  def vmap_grad(params, state, data_batch, rng, is_training):
    # The non gradient part gets computed and averaged first.
    value2 = fn2(params, state, data_batch, rng, is_training)

    # We pmean over all devices. This results in a lower variance
    # estimates but keeps the unbiased property of the estimator since
    # it avoids backpropagating through a gradient operation.
    if use_pmap_on_stop_grad_tree:
      value2 = jax.lax.pmean(value2, axis_name='i')

    grad_fn = _select_grad(jax.grad(dot_prod, argnums=1), grad_var)
    return grad_fn(value2, params, state, data_batch, rng, is_training)

  return vmap_grad


def two_data_estimate_grad_fn(fn1, fn2, grad_var='disc'):
  def estimate(
      params, state, data_batch1, data_batch2, rng1, rng2, is_training):
    value1 = fn1(params, state, data_batch1, rng1, is_training)
    value2 = fn2(params, state, data_batch2, rng2, is_training)
    value2 = jax.lax.stop_gradient(value2)
    return utils.tree_dot_product(value1, value2)

  return _select_grad(jax.grad(estimate), grad_var)
