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
"""Test implementations of regularizer estimates on 1 device."""


from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import numpy as np

from dd_two_player_games import gan
from dd_two_player_games import losses
from dd_two_player_games import regularizer_estimates


class SimpleMLP(hk.nets.MLP):

  def __init__(self, depth, hidden_size, out_dim, name='SimpleMLP'):
    output_sizes = [hidden_size] * depth + [out_dim]
    super(SimpleMLP, self).__init__(output_sizes, name=name)

  def __call__(self, inputs, is_training=True):
    del is_training
    return super(SimpleMLP, self).__call__(inputs)


class MixtureOfGaussiansDataset():

  def __init__(self, batch_size):
    self.batch_size = batch_size
    self.mog_mean = np.array([
        [1.50, 1.50],
        [1.50, 0.50],
        [1.50, -0.50],
        [1.50, -1.50],
        [0.50, 1.50],
        [0.50, 0.50],
        [0.50, -0.50],
        [0.50, -1.50],
        [-1.50, 1.50],
        [-1.50, 0.50],
        [-1.50, -0.50],
        [-1.50, -1.50],
        [-0.50, 1.50],
        [-0.50, 0.50],
        [-0.50, -0.50],
        [-0.50, -1.50]])

  def __iter__(self):
    return self

  def __next__(self):
    temp = np.tile(self.mog_mean, (self.batch_size // 16 + 1, 1))
    mus = temp[0:self.batch_size, :]
    return mus + 0.02 * np.random.normal(size=(self.batch_size, 2))


def init_gan(gan_module, data_batch):
  init_params_rng = jax.random.PRNGKey(0)
  rng = jax.random.PRNGKey(4)
  # Params and data need to be over the number of batches.
  params, state = gan_module.initial_params(init_params_rng, data_batch)
  return params, state, data_batch, rng


REGULARIZER_TUPLES = [
    ('disc_loss_fn_disc_grads', 'disc_loss_fn_disc_grads', 'disc'),
    ('gen_loss_fn_gen_grads', 'gen_loss_fn_gen_grads', 'disc'),
    ('disc_loss_fn_gen_grads', 'gen_loss_fn_gen_grads', 'disc'),
    ('disc_loss_fn_disc_grads', 'disc_loss_fn_disc_grads', 'gen'),
    ('gen_loss_fn_disc_grads', 'disc_loss_fn_disc_grads', 'gen'),
    ('gen_loss_fn_gen_grads', 'gen_loss_fn_gen_grads', 'gen')]


IS_TRAINING_VALUES = [True, False]


DEFAULT_TEST_INPUT = [
    x + (y,) for x in REGULARIZER_TUPLES for y in IS_TRAINING_VALUES]  # pylint: disable=g-complex-comprehension


class RegularizerEstimatesTest(parameterized.TestCase):

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def test_biased_estimate_implementation_consistency_disc_norm(
      self, fn1_name, fn2_name, grad_var, is_training):
    batch_size = 128
    dataset = MixtureOfGaussiansDataset(batch_size=batch_size)

    data_batch = next(dataset)

    players_hk_tuple = gan.GANTuple(disc=SimpleMLP, gen=SimpleMLP)

    losses_tuple = gan.GANTuple(
        disc=losses.discriminator_goodfellow_loss,
        gen=losses.generator_goodfellow_loss)

    gan_module = gan.GAN(
        players_hk=players_hk_tuple, losses=losses_tuple,
        penalties=gan.GANTuple(disc=None, gen=None),
        player_param_transformers=gan.GANTuple(disc=None, gen=None),
        players_kwargs=gan.GANTuple(
            disc={'depth': 4, 'out_dim': 1, 'hidden_size': 25},
            gen={'depth': 4, 'out_dim': 2, 'hidden_size': 25},),
        num_latents=16)

    fn1 = getattr(gan_module, fn1_name)
    fn2 = getattr(gan_module, fn2_name)

    grad_fn1 = regularizer_estimates.biased_estimate_general_grad_fn(
        fn1, fn2, False, grad_var=grad_var)

    grad_fn2 = regularizer_estimates.biased_estimate_grad_fn(
        fn1, fn2, grad_var=grad_var)

    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan(gan_module, data_batch)

    output1 = grad_fn1(
        devices_params, devices_state, devices_data_batch, devices_rng,
        is_training)

    output2 = grad_fn2(
        devices_params, devices_state, devices_data_batch, devices_rng,
        is_training)

    chex.assert_trees_all_close(output1, output2, rtol=1e-4, atol=1e-2)


if __name__ == '__main__':
  absltest.main()
