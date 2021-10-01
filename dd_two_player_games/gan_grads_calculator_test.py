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
"""Test implementations of GAN grad calculator."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from dd_two_player_games import drift_utils
from dd_two_player_games import gan
from dd_two_player_games import gan_grads_calculator
from dd_two_player_games import losses
from dd_two_player_games import regularizer_estimates


class DiracDiscriminator(hk.Linear):
  """DiracGAN discriminator module."""

  def __init__(self, init_value=0.1):
    super(DiracDiscriminator, self).__init__(
        output_size=1, with_bias=False,
        w_init=hk.initializers.Constant(init_value),
        name='DiracDiscriminator')

  def __call__(self, inputs, is_training=False):
    del is_training
    return super(DiracDiscriminator, self).__call__(inputs)


class DiracGenerator(hk.Module):
  """DiracGAN generator module."""

  def __init__(self, init_value=0.1):
    super(DiracGenerator, self).__init__(name='DiracGenerator')
    self.init_value = init_value

  def __call__(self, x, is_training=False):
    del x, is_training
    w = hk.get_parameter(
        'w', [1, 1], jnp.float32,
        init=hk.initializers.Constant(self.init_value))
    return w


def l_prime(x):
  return 1 / (1 + np.exp(-x))


def l_second(x):
  return np.exp(-x)/ (1 + np.exp(-x))**2


DISC_INIT_VALUES = [-0.5, -0.1, 0.1, 0.5]
GEN_INIT_VALUES = [-0.5, -0.1, 0.1, 0.5]

DEFAULT_TEST_INPUT = list(itertools.product(DISC_INIT_VALUES, GEN_INIT_VALUES))


class GANGradsCalculatorTest(parameterized.TestCase):

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def testDiracGANGradsNoExplicitReg(self, disc_init_value, gen_init_value):
    """Tests that DiracGAN gradients are as expected."""
    players_hk_tuple = gan.GANTuple(
        disc=DiracDiscriminator, gen=DiracGenerator)

    losses_tuple = gan.GANTuple(
        disc=losses.discriminator_goodfellow_loss,
        gen=losses.generator_saturating_loss)

    gan_module = gan.GAN(
        players_hk=players_hk_tuple, losses=losses_tuple,
        penalties=gan.GANTuple(disc=None, gen=None),
        player_param_transformers=gan.GANTuple(disc=None, gen=None),
        players_kwargs=gan.GANTuple(
            disc={'init_value': disc_init_value},
            gen={'init_value': gen_init_value}),
        num_latents=1)

    grad_calculator = gan_grads_calculator.GANGradsCalculator(
        gan_module, estimator_fn=regularizer_estimates.biased_estimate_grad_fn,
        use_pmean=False)

    zero_coeffs = drift_utils.PlayerRegularizationTerms(
        self_norm=0, other_norm=0, other_dot_prod=0)
    coeffs = gan.GANTuple(disc=zero_coeffs, gen=zero_coeffs)

    # Initial rng, not used when creating samples so OK to share between
    # parameters and generator.
    init_rng = jax.random.PRNGKey(0)
    # a toy cyclic iterator, not used.
    dataset = itertools.cycle([jnp.array([[0.]])])

    init_params, state = gan_module.initial_params(init_rng, next(dataset))
    params = init_params

    data_batch = next(dataset)
    disc_grads_values = grad_calculator.disc_grads(
        params, state, data_batch, init_rng, True, coeffs)

    # assert the values of the discriminator
    disc_param = jax.tree_leaves(params.disc)[0]
    gen_param = jax.tree_leaves(params.gen)[0]

    expected_disc_grads_values = l_prime(disc_param * gen_param) * gen_param

    chex.assert_trees_all_close(
        jax.tree_leaves(disc_grads_values)[0],
        expected_disc_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

    gen_grads_values = grad_calculator.gen_grads(
        params, state, data_batch, init_rng, True, coeffs)

    expected_gen_grads_values = - l_prime(disc_param * gen_param) * disc_param

    chex.assert_trees_all_close(
        jax.tree_leaves(gen_grads_values)[0],
        expected_gen_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def testDiracGANGradsExplicitRegSelfNorm(
      self, disc_init_value, gen_init_value):
    """Tests that DiracGAN gradients are as expected."""
    players_hk_tuple = gan.GANTuple(
        disc=DiracDiscriminator, gen=DiracGenerator)

    losses_tuple = gan.GANTuple(
        disc=losses.discriminator_goodfellow_loss,
        gen=losses.generator_saturating_loss)

    gan_module = gan.GAN(
        players_hk=players_hk_tuple, losses=losses_tuple,
        penalties=gan.GANTuple(disc=None, gen=None),
        player_param_transformers=gan.GANTuple(disc=None, gen=None),
        players_kwargs=gan.GANTuple(
            disc={'init_value': disc_init_value},
            gen={'init_value': gen_init_value}),
        num_latents=1)

    grad_calculator = gan_grads_calculator.GANGradsCalculator(
        gan_module, estimator_fn=regularizer_estimates.biased_estimate_grad_fn,
        use_pmean=False)

    self_norm_reg_value = 10.
    self_norm = drift_utils.PlayerRegularizationTerms(
        self_norm=self_norm_reg_value, other_norm=0, other_dot_prod=0)
    coeffs = gan.GANTuple(disc=self_norm, gen=self_norm)

    # Initial rng, not used when creating samples so OK to share between
    # parameters and generator.
    init_rng = jax.random.PRNGKey(0)
    # a toy cyclic iterator, not used.
    dataset = itertools.cycle([jnp.array([[0.]])])

    init_params, state = gan_module.initial_params(init_rng, next(dataset))
    params = init_params

    data_batch = next(dataset)
    disc_grads_values = grad_calculator.disc_grads(
        params, state, data_batch, init_rng, True, coeffs)

    # assert the values of the discriminator
    disc_param = jax.tree_leaves(params.disc)[0]
    gen_param = jax.tree_leaves(params.gen)[0]
    param_prod = disc_param * gen_param

    # loss grad
    expected_disc_grads_values = l_prime(param_prod) * gen_param
    # explicit regularisation grad
    expected_disc_grads_values += (
        self_norm_reg_value * gen_param ** 3 * l_prime(
            param_prod) * l_second(param_prod))

    chex.assert_trees_all_close(
        jax.tree_leaves(disc_grads_values)[0],
        expected_disc_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

    gen_grads_values = grad_calculator.gen_grads(
        params, state, data_batch, init_rng, True, coeffs)

    expected_gen_grads_values = - l_prime(param_prod) * disc_param
    expected_gen_grads_values += (
        self_norm_reg_value * disc_param ** 3 * l_prime(
            param_prod) * l_second(param_prod))

    chex.assert_trees_all_close(
        jax.tree_leaves(gen_grads_values)[0],
        expected_gen_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def testDiracGANGradsExplicitRegOtherNorm(
      self, disc_init_value, gen_init_value):
    """Tests that DiracGAN gradients are as expected."""
    players_hk_tuple = gan.GANTuple(
        disc=DiracDiscriminator, gen=DiracGenerator)

    losses_tuple = gan.GANTuple(
        disc=losses.discriminator_goodfellow_loss,
        gen=losses.generator_saturating_loss)

    gan_module = gan.GAN(
        players_hk=players_hk_tuple, losses=losses_tuple,
        penalties=gan.GANTuple(disc=None, gen=None),
        player_param_transformers=gan.GANTuple(disc=None, gen=None),
        players_kwargs=gan.GANTuple(
            disc={'init_value': disc_init_value},
            gen={'init_value': gen_init_value}),
        num_latents=1)

    grad_calculator = gan_grads_calculator.GANGradsCalculator(
        gan_module, estimator_fn=regularizer_estimates.biased_estimate_grad_fn,
        use_pmean=False)

    other_norm_reg_value = 10.
    self_norm = drift_utils.PlayerRegularizationTerms(
        self_norm=0, other_norm=other_norm_reg_value, other_dot_prod=0)
    coeffs = gan.GANTuple(disc=self_norm, gen=self_norm)

    # Initial rng, not used when creating samples so OK to share between
    # parameters and generator.
    init_rng = jax.random.PRNGKey(0)
    # a toy cyclic iterator, not used.
    dataset = itertools.cycle([jnp.array([[0.]])])

    init_params, state = gan_module.initial_params(init_rng, next(dataset))
    params = init_params

    data_batch = next(dataset)
    disc_grads_values = grad_calculator.disc_grads(
        params, state, data_batch, init_rng, True, coeffs)

    # assert the values of the discriminator
    disc_param = jax.tree_leaves(params.disc)[0]
    gen_param = jax.tree_leaves(params.gen)[0]
    param_prod = disc_param * gen_param

    # loss grad
    expected_disc_grads_values = l_prime(param_prod) * gen_param
    # explicit regularisation grad
    expected_disc_grads_values += other_norm_reg_value * (
        disc_param * l_prime(param_prod) ** 2
        + disc_param ** 2 * gen_param * l_prime(
            param_prod) * l_second(param_prod))

    chex.assert_trees_all_close(
        jax.tree_leaves(disc_grads_values)[0],
        expected_disc_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

    gen_grads_values = grad_calculator.gen_grads(
        params, state, data_batch, init_rng, True, coeffs)

    expected_gen_grads_values = - l_prime(param_prod) * disc_param
    expected_gen_grads_values += other_norm_reg_value * (
        gen_param * l_prime(param_prod) ** 2
        + gen_param ** 2 * disc_param * l_prime(
            param_prod) * l_second(param_prod))

    chex.assert_trees_all_close(
        jax.tree_leaves(gen_grads_values)[0],
        expected_gen_grads_values,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)


if __name__ == '__main__':
  absltest.main()
