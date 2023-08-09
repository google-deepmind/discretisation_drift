"""Copyright 2023 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions
and limitations under the License.


The code below provides the implementation for the experiments used in
"On a continuous time model of gradient descent dynamics and instability in
deep learning", TMLR 2023 when using DAL instead of vanilla gradient descent.
"""

import jax
import jax.numpy as jnp


class DriftAdaptiveLearningRateScaler(object):
  r"""Class which scales the gradient proportionally to the norm of the drift.

  This class implements the DAL algorithm here: https://arxiv.org/abs/2302.01952

  Note: the user can specify either:
    * grad_fn: the gradient function of the model.
          Can be obtained using jax.grad.
    * hvp_fn: the hessian_vector_product of the model. We provide an
      implementation of hvp below.

    The reason we do not compute these here is that for large batch sizes,
    the user might want to accumulate gradients over batches before passing it
    here. We do that in the full batch case implemented in the paper, but not
    for stochastic gradient descent. Similarly, if running over multiple devices
    via `pmap`, the user has to ensure that grad_fn and hvp_fn have been wrapped
    in `pmap` accordingly before creating this object.

  scaling_power: denotes the `p` in the paper. If 0, vanilla SGD is used.

  Note: static_lr below has no effect if DAL is used, that is, if scaling_power
    is not 0. We support it here so that we can also implement vanilla SGD
    with the same optimiser.

  We know that the total drift of gradient descent in one area is given by
  lr * || H(\theta') g(\theta')|| for a certain \theta' in the neighborhood of
  the current parameters \theta. We use this to get an idea of the size of the
  drift in certain areas, by using lr * ||H(\theta) g(\theta)|| as a guide.

  We then scale the size of the gradient by 1/ (lr * ||H(\theta) g(\theta)||^p),
  where p is a hyperparameter chosen by the user. For p=0, no scaling occurs.
  For p = 1, there is a scaling proportional to Hg.
  """

  def __init__(
      self, hvp_fn, grad_fn,
      scaling_power,
      static_lr,
      max_lr_scale=5,
      scaling_start=None, scaling_stop=None, always_log_h_g_norm=True):
    self.hvp_fn = hvp_fn
    self.grad_fn = grad_fn
    self.scaling_power = scaling_power
    self.static_lr = static_lr
    self.max_lr_scale = max_lr_scale
    self.scaling_start = scaling_start if scaling_start else -1
    self.scaling_stop = scaling_stop if scaling_stop else np.inf
    self.always_log_h_g_norm = always_log_h_g_norm

    if hvp_fn and grad_fn:
      raise ValueError(
          'Either use hvp_fn or grad_fn as an argument for '
          'DriftAdaptiveLearningRateScaler')

  def _h_g(self, params, grads, *grad_fn_args):
    if self.hvp_fn:
      return self.hvp_fn(params, grads, *grad_fn_args)

    return hg_finite_differences_approx(self.grad_fn)(
        params, grads, *grad_fn_args)

  def scale_grads(self, grads, params, iteration, *grad_fn_args):
    """Compute (scaled) gradients at given parameters."""
    # Fixed coefficient (not the result of a jax operation) such that
    # multioptimiser knows the number of steps and returns all intermediate
    # results for static learning rates.
    h_g = self._h_g(params, grads, *grad_fn_args)
    h_g_norm = jnp.sqrt(utils.tree_square_norm(h_g))

    if self.scaling_power == 0:
      # No learning rate scaling.
      return grads, 1., h_g_norm if self.always_log_h_g_norm else -1.

    grad_norm = jnp.sqrt(utils.tree_square_norm(grads))
    max_scale = self.max_lr_scale / self.static_lr

    lr_scale = self._adaptive_lr_multiplier(
        h_g_norm/grad_norm, iteration, max_scale)
    grads = utils.tree_mul(grads, lr_scale)

    return grads, lr_scale, h_g_norm

  def _adaptive_lr_multiplier(self, val, iteration, max_scale):
    def scale(v):
      denom = self.static_lr  * jnp.power(v, self.scaling_power) /2.
      return jnp.clip(1 /  denom, 0., max_scale)

    return jax.lax.cond(
        (iteration > self.scaling_start) & (iteration < self.scaling_stop),
        scale,
        lambda _: 1., val)  # No learning rate scaling outside the interval.



def hg_finite_differences_approx(grad_fn, epsilon=None):
  r"""Estimate Hg via finite differences.

   Hg \approx (
      \nabla_{\theta}E(theta + epsilon) - \nabla_{\theta}E(theta)) /epsilon

   Approximation used by https://arxiv.org/abs/2109.14119.

  Args:
    grad_fn: The gradient function (backward pass).
    epsilon: The offset. If None, is set to 0.01 * ||\nabla_{\theta}E(theta)||.

  Returns:
    A function which given a set of parameters and gradients returns the
      approximation above to the gradient norm.
  """
  def finite_differences_fn(params, grads, *args):
    grad_sq_norm = tree_square_norm(grads)
    e = epsilon if epsilon else 0.01 * jax.lax.rsqrt(grad_sq_norm)
    delta_params = tree_add(params, tree_mul(grads, e))
    _, grad_at_delta_params = grad_fn(delta_params, *args)
    return tree_mul(tree_diff(grad_at_delta_params, grads), 1./e)

  return finite_differences_fn


def reduce_sum_tree(tree):
  tree = jax.tree_util.tree_map(jnp.sum, tree)
  return jax.tree_util.tree_reduce(jnp.add, tree)


def tree_square_norm(tree):
  return reduce_sum_tree(jax.tree_map(jnp.square, tree))


def tree_mul(tree, constant):
  return jax.tree_map(lambda x: x * constant, tree)


def tree_add(tree1, tree2):
  return jax.tree_map(lambda x, y: x + y, tree1, tree2)


def tree_diff(tree1, tree2):
  return jax.tree_map(lambda x, y: x - y, tree1, tree2)


def hvp(loss_fn, params, v, *args):
  """Computes the hessian vector product Hv.

  This implementation uses forward-over-reverse mode for computing the hvp.

  Args:
    loss_fn: function computing the loss with signature
      loss(params, batch).
    params: pytree for the parameters of the model.
    v: pytree of the same structure as params.
    *args: other arguments to pass to loss_fn.

  Returns:
    hvp: array of shape [num_params] equal to Hv where H is the hessian.
  """
  def grad_fn(p):
    return jax.grad(loss_fn)(p, *args)

  return jax.jvp(grad_fn, [params], [v])[1]
