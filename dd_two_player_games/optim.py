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
"""Optimizers and related utilities."""

from dd_two_player_games import utils


class RungeKutta(object):
  """Runge Kutta optimizer."""

  def __init__(self, grad_fn, runge_kutta_order, same_args_to_grad_calls=False):
    """Initializes the RK object.

    Args:
      grad_fn: Gradient function. We assume the gradient function already
        has called `pmean` if that is required.
      runge_kutta_order: The RK order. Supported 1, 2, 4. Setting RK to 1
      is the same as SGD.
      same_args_to_grad_calls: whether or not to use the same arguments (ie
        data batch, or rng) for each call to grad inside the RK computation.
    """
    self.runge_kutta_order = runge_kutta_order
    self.same_args_to_grad_calls = same_args_to_grad_calls
    self.grad_fn = grad_fn

    if runge_kutta_order == 1:
      self.grad_weights = [1.]
      self.step_scale = []
    if runge_kutta_order == 2:
      self.grad_weights = [0.5, 0.5]
      self.step_scale = [1.]
    elif runge_kutta_order == 4:
      self.grad_weights = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
      self.step_scale = [0.5, 0.5, 1.]

  def grad(self, params, step_size, *grad_fn_args):
    """Compute gradients at given parameters."""
    def select_args(i):
      if self.same_args_to_grad_calls:
        # Use the first batch / set of arguments for every grad call.
        return tuple([x[0] for x in grad_fn_args])
      return tuple([x[i] for x in grad_fn_args])

    step_grad, aux = self.grad_fn(params, *select_args(0))
    grad = utils.tree_mul(step_grad, self.grad_weights[0])
    for i in range(self.runge_kutta_order - 1):
      step_params = utils.tree_add(
          params,
          utils.tree_mul(step_grad, -1 * self.step_scale[i] * step_size))
      step_grad, _ = self.grad_fn(step_params, *select_args(i+1))
      grad = utils.tree_add(
          grad,
          utils.tree_mul(step_grad, self.grad_weights[i+1]))

    return grad, aux
