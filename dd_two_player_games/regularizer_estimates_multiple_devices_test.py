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
"""Test implementations of regularizer estimates on multiple devices."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from dd_two_player_games import gan
from dd_two_player_games import losses
from dd_two_player_games import regularizer_estimates


# Use CPU for testing since we use chex to set multiple devices.
jax.config.update('jax_platform_name', 'cpu')


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


def pmap_and_average(fn):
  def mapped_fn(*args):
    return jax.lax.pmean(fn(*args), axis_name='i')
  return jax.pmap(mapped_fn, axis_name='i', static_broadcasted_argnums=4)


def bcast_local_devices(value):
  """Broadcasts an object to all local devices."""
  devices = jax.local_devices()
  return jax.tree_map(
      lambda v: jax.device_put_sharded(len(devices) * [v], devices), value)


def init_gan_multiple_devices(gan_module, num_devices, data_batch, pmap=True):
  batch_size = data_batch.shape[0]

  if pmap:
    reshaped_data_batch = np.reshape(
        data_batch, (num_devices, batch_size // num_devices, -1))
    init_gan = jax.pmap(gan_module.initial_params, axis_name='i')
  else:
    reshaped_data_batch = data_batch
    init_gan = gan_module.initial_params

  if pmap:
    init_params_rng = bcast_local_devices(jax.random.PRNGKey(0))
    # We use the same rng - note, this will not be the case in practice
    # but we have no other way of controlling the samples.
    rng = bcast_local_devices(jax.random.PRNGKey(4))
  else:
    init_params_rng = jax.random.PRNGKey(0)
    rng = jax.random.PRNGKey(4)

  # Params and data need to be over the number of batches.
  params, state = init_gan(init_params_rng, reshaped_data_batch)

  return params, state, reshaped_data_batch, rng


# Simple estimate that can be used for the 1 device computation of
# unbiased estimates.
def unbiased_estimate_1_device(fn1, fn2, grad_var='disc'):
  two_data_fn = regularizer_estimates.two_data_estimate_grad_fn(
      fn1, fn2, grad_var)
  def vmap_grad(params, state, data_batch, rng, is_training):
    data1, data2 = jnp.split(data_batch, [int(data_batch.shape[0]/2),], axis=0)

    # Split rngs so that we can vmap the operations and obtained an unbiased
    # estimate. Note: this means that we will obtain a different sample
    # for each vmap, but due to the split we will also obtain a different
    # set of samples than those obtained in the forward pass.
    rng1, rng2 = jax.random.split(rng, 2)

    return two_data_fn(params, state, data1, data2, rng1, rng2, is_training)

  return vmap_grad


NUM_DEVICES = 4

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


class RegularizerEstimatesMultipleEstimatesTest(parameterized.TestCase):

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def test_same_result_multiple_devices_biased(
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

    # Multiple devices
    chex.set_n_cpu_devices(NUM_DEVICES)

    grad_fn = regularizer_estimates.biased_estimate_general_grad_fn(
        fn1, fn2, True, grad_var=grad_var)
    pmap_grad_fn = pmap_and_average(grad_fn)
    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=True)

    multiple_device_output = pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    # No pmap
    no_pmap_grad_fn = regularizer_estimates.biased_estimate_general_grad_fn(
        fn1, fn2, False, grad_var=grad_var)

    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=False)
    one_device_output = no_pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    chex.assert_trees_all_close(
        jax.tree_map(lambda x: x[0], multiple_device_output),
        one_device_output,
        rtol=1e-3,
        atol=6e-2,
        ignore_nones=True)

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def test_same_result_multiple_devices_unbiased_consistency(
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

    # Multiple devices
    chex.set_n_cpu_devices(NUM_DEVICES)

    grad_fn = regularizer_estimates.unbiased_estimate_fn_lowest_variance(
        fn1, fn2, grad_var=grad_var)
    pmap_grad_fn = pmap_and_average(grad_fn)
    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=True)

    multiple_device_output = pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    # No pmap
    no_pmap_grad_fn = regularizer_estimates.unbiased_estimate_fn_different_device_results(
        fn1, fn2, grad_var=grad_var)

    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=False)
    one_device_output = no_pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    chex.assert_trees_all_close(
        jax.tree_map(lambda x: x[0], multiple_device_output),
        one_device_output,
        rtol=1e-3,
        atol=5e-2)

  @parameterized.parameters(DEFAULT_TEST_INPUT)
  def test_same_result_multiple_devices_unbiased(
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

    # Multiple devices
    chex.set_n_cpu_devices(NUM_DEVICES)

    grad_fn = regularizer_estimates.unbiased_estimate_fn_lowest_variance(
        fn1, fn2, grad_var=grad_var)
    pmap_grad_fn = pmap_and_average(grad_fn)
    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=True)

    multiple_device_output = pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    # No pmap
    no_pmap_grad_fn = unbiased_estimate_1_device(fn1, fn2, grad_var=grad_var)

    (devices_params, devices_state,
     devices_data_batch, devices_rng) = init_gan_multiple_devices(
         gan_module, NUM_DEVICES, data_batch, pmap=False)
    one_device_output = no_pmap_grad_fn(
        devices_params, devices_state, devices_data_batch,
        devices_rng, is_training)

    chex.assert_trees_all_close(
        jax.tree_map(lambda x: x[0], multiple_device_output),
        one_device_output,
        rtol=1e-3,
        atol=5e-2)


if __name__ == '__main__':
  absltest.main()
