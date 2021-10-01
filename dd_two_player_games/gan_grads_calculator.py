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
"""Class which gives gan grads with explicit regularization."""

import functools
import jax
import jax.numpy as jnp

from dd_two_player_games import drift_utils
from dd_two_player_games import gan
from dd_two_player_games import utils


def explicit_regularization_grads(
    regularizer_fns, coeffs,
    params, state, data_batch, rng, is_training, attr_name):
  """Accumulate explicit regularization gradients."""
  player_regularizer_fns = getattr(regularizer_fns, attr_name)
  player_coeffs = getattr(coeffs, attr_name)

  assert isinstance(
      player_regularizer_fns, drift_utils.PlayerRegularizationTerms)
  assert isinstance(
      player_coeffs, drift_utils.PlayerRegularizationTerms)

  reg_args = (params, state, data_batch, rng, is_training)
  accumulator = jax.tree_map(jnp.zeros_like, getattr(params, attr_name))

  for grad_reg_fn, coeff in zip(player_regularizer_fns, player_coeffs):
    accumulator = utils.add_trees_with_coeff(
        acc=accumulator,
        mul=grad_reg_fn(*reg_args),
        coeff=coeff)

  return accumulator, utils.any_non_zero(player_coeffs)


class GANGradsCalculator(object):
  """Utility class to compute GAN gradients with explicit regularization."""

  def __init__(self, gan_model, estimator_fn, use_pmean=True):
    super(GANGradsCalculator, self).__init__()

    self._gan = gan_model
    self._estimator_fn = estimator_fn
    self._use_pmean = use_pmean

    self._maybe_pmean = jax.lax.pmean if use_pmean else lambda x, axis_name: x

  @property
  def regularisation_grad_fns(self):
    """Create the regularisation functions."""
    disc_reg_grad_fns = drift_utils.PlayerRegularizationTerms(
        self_norm=self.disc_grads_disc_norm,
        other_norm=self.disc_grads_gen_norm,
        other_dot_prod=self.disc_grads_gen_dot_prod)
    gen_reg_grad_fns = drift_utils.PlayerRegularizationTerms(
        self_norm=self.gen_grads_gen_norm,
        other_norm=self.gen_grads_disc_norm,
        other_dot_prod=self.gen_grads_disc_dot_prod)
    return gan.GANTuple(disc=disc_reg_grad_fns, gen=gen_reg_grad_fns)

  @property
  def disc_grads_disc_norm(self):
    return self._estimator_fn(
        self._gan.disc_loss_fn_disc_grads,
        self._gan.disc_loss_fn_disc_grads,
        'disc')

  @property
  def disc_grads_gen_norm(self):
    return self._estimator_fn(
        self._gan.gen_loss_fn_gen_grads,
        self._gan.gen_loss_fn_gen_grads,
        'disc')

  @property
  def disc_grads_gen_dot_prod(self):
    # Note: order matters due to the stop grad.
    return self._estimator_fn(
        self._gan.disc_loss_fn_gen_grads,
        self._gan.gen_loss_fn_gen_grads,
        'disc')

  @property
  def gen_grads_gen_norm(self):
    return self._estimator_fn(
        self._gan.gen_loss_fn_gen_grads,
        self._gan.gen_loss_fn_gen_grads,
        'gen')

  @property
  def gen_grads_disc_norm(self):
    return self._estimator_fn(
        self._gan.disc_loss_fn_disc_grads,
        self._gan.disc_loss_fn_disc_grads,
        'gen')

  @property
  def gen_grads_disc_dot_prod(self):
    # Note: order matters due to the stop grad.
    return self._estimator_fn(
        self._gan.gen_loss_fn_disc_grads,
        self._gan.disc_loss_fn_disc_grads,
        'gen')

  @functools.partial(jax.jit, static_argnums=(0, 5, 6))
  def disc_grads(
      self, params, state, data_batch, rng_disc, is_training, coeffs):
    """Compute discriminator gradients."""
    disc_loss_grads, disc_gan_loss_aux = jax.grad(
        self._gan.disc_loss, has_aux=True)(
            params, state, data_batch, rng_disc, is_training)
    disc_loss_grads = disc_loss_grads.disc

    disc_reg_grads, non_zero_coeff = self.disc_explicit_regularization_grads(
        params, state, data_batch, rng_disc, is_training, coeffs)

    disc_grads = utils.add_trees_with_coeff(
        acc=disc_loss_grads,
        mul=disc_reg_grads,
        coeff=non_zero_coeff)
    disc_grads = self._maybe_pmean(disc_grads, axis_name='i')
    return disc_grads, disc_gan_loss_aux

  @functools.partial(jax.jit, static_argnums=(0, 5, 6))
  def gen_grads(
      self, params, state, data_batch, rng_gen, is_training, coeffs):
    """Compute generator gradients."""
    gen_loss_grads, gen_gan_loss_aux = jax.grad(
        self._gan.gen_loss, has_aux=True)(
            params, state, data_batch, rng_gen, is_training)
    gen_loss_grads = gen_loss_grads.gen

    gen_reg_grads, non_zero_coeff = self.gen_explicit_regularization_grads(
        params, state, data_batch, rng_gen, is_training, coeffs)

    gen_grads = utils.add_trees_with_coeff(
        acc=gen_loss_grads,
        mul=gen_reg_grads,
        coeff=non_zero_coeff)

    gen_grads = self._maybe_pmean(gen_grads, axis_name='i')
    return gen_grads, gen_gan_loss_aux

  def disc_explicit_regularization_grads(
      self, params, state, data_batch, rng_disc, is_training, coeffs):
    return explicit_regularization_grads(
        self.regularisation_grad_fns, coeffs,
        params, state, data_batch, rng_disc, is_training, 'disc')

  def gen_explicit_regularization_grads(
      self, params, state, data_batch, rng_gen, is_training, coeffs):
    return explicit_regularization_grads(
        self.regularisation_grad_fns, coeffs,
        params, state, data_batch, rng_gen, is_training, 'gen')

  def both_player_grads(
      self, params, state, disc_data_batch, gen_data_batch,
      rng_disc, rng_gen, is_training, exp_reg_coeffs):
    """Compute the gradients for both players and return them as a GANTuple."""
    disc_grads, disc_aux = self.disc_grads(
        params, state, disc_data_batch, rng_disc, is_training,
        exp_reg_coeffs)
    gen_grads, gen_aux = self.gen_grads(
        params, state, gen_data_batch, rng_gen, is_training,
        exp_reg_coeffs)
    grads = gan.GANTuple(disc=disc_grads, gen=gen_grads)
    aux = gan.GANTuple(disc=disc_aux, gen=gen_aux)
    return grads, aux
