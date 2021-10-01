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
"""Loss functions used for discriminator and generator training."""

import collections

import jax
import jax.numpy as jnp

from dd_two_player_games import utils


LossOutput = collections.namedtuple(
    'LossOutput', ['total', 'components'])


def binary_cross_entropy_loss(logits, targets):
  """Binary cross entropy loss between logit and targets.

  Computes:
    - (targets * log(sigmoid(x)) + (1 - targets) * log(1 - sigmoid(x)))

    If targets = 1: log(sigmoid(x)) = - log(1 + exp(-x))
    If targets = 0: log(1- sigmoid(x)) = - log(1 + exp(x))

  Args:
    logits: Logits jnp array.
    targets: Binary jnp array. Same shape as `logits`.

  Returns:
    A jnp array with 1 element, the value of the binary cross entropy loss
    given the input logits and targets.
  """
  assert logits.shape == targets.shape
  neg_log_probs = jnp.logaddexp(0., jnp.where(targets, -1., 1.) * logits)
  assert neg_log_probs.shape == (logits.shape[0], 1), neg_log_probs.shape
  return jnp.mean(neg_log_probs)


def square_loss(logits, targets):
  return jnp.mean((logits - targets) ** 2)


def square_loss_with_sigmoid_logits(logits, targets):
  return jnp.mean((jax.nn.sigmoid(logits) - targets) ** 2)


### ************* Player losses.*************


# Discriminator and generator losses given by proper scoring rules.
def discriminator_scoring_rule_loss(scoring_rule_fn):
  """Constructs discriminator loss from scoring rule."""
  def discrimiantor_loss_fn(
      discriminator_real_data_outputs, discriminator_sample_outputs):
    samples_loss = scoring_rule_fn(
        discriminator_sample_outputs,
        jnp.zeros_like(discriminator_sample_outputs))
    data_loss = scoring_rule_fn(
        discriminator_real_data_outputs,
        jnp.ones_like(discriminator_real_data_outputs))
    loss = data_loss + samples_loss
    loss_components = {
        'disc_data_loss': data_loss,
        'disc_samples_loss': samples_loss}
    return LossOutput(loss, loss_components)
  return discrimiantor_loss_fn


def generator_scoring_rule_fn_loss(scoring_rule_fn):
  """Constructs generator loss from scoring rule."""
  def generator_loss_fn(
      discriminator_real_data_outputs, discriminator_sample_outputs):
    del discriminator_real_data_outputs
    loss = - scoring_rule_fn(
        discriminator_sample_outputs,
        jnp.zeros_like(discriminator_sample_outputs))
    return LossOutput(loss, {'gen_samples_loss': loss})
  return generator_loss_fn


def discriminator_goodfellow_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Binary discriminator loss."""
  return discriminator_scoring_rule_loss(binary_cross_entropy_loss)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def generator_goodfellow_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Generator loss proposed by the original GAN paper.

  This generator loss corresponds to -log(D) (used in vanilla GAN).

  The loss proposed by the original GAN paper as a substitute for the min-max
  loss, since it provides better gradients. Details at:
  https://arxiv.org/abs/1406.2661.

  Args:
    discriminator_real_data_outputs: the output from a discriminator net on
      real data.
    discriminator_sample_outputs: the output from a discriminator net on
      generated data.

  Returns:
    The non saturating loss of the generator (a jnp.array): - log (D).
  """
  del discriminator_real_data_outputs  # unused
  loss = binary_cross_entropy_loss(
      discriminator_sample_outputs, jnp.ones_like(discriminator_sample_outputs))
  return LossOutput(loss, {'gen_samples_loss': loss})


def generator_saturating_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Saturating generator loss proposed by the original GAN paper.

  This loss corresponds to the min-max version of the game, and provides
  worse gradients early on in training (when the generator is not performing
  well).

  Args:
    discriminator_real_data_outputs: the output from a discriminator net on
      real data.
    discriminator_sample_outputs: the output from a discriminator net on
      generated data.

  Returns:
    The saturating loss of the generator (a jnp.array): log (1 - D).
  """
  return generator_scoring_rule_fn_loss(binary_cross_entropy_loss)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def discriminator_square_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Binary discriminator loss."""
  # Real data is classified as 1, fake data classified as 0.
  return discriminator_scoring_rule_loss(square_loss)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def generator_square_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  return generator_scoring_rule_fn_loss(square_loss)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def discriminator_square_loss_with_sigmoid(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Binary discriminator loss."""
  # Real data is classified as 1, fake data classified as 0.
  return discriminator_scoring_rule_loss(square_loss_with_sigmoid_logits)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def generator_square_loss_with_sigmoid(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  return generator_scoring_rule_fn_loss(square_loss_with_sigmoid_logits)(
      discriminator_real_data_outputs, discriminator_sample_outputs)


def wasserstein_discriminator_loss(
    discriminator_real_data_outputs, discriminator_sample_outputs):
  """Discriminator loss according to the Wasserstein loss."""
  data_loss = jnp.mean(- discriminator_real_data_outputs)
  samples_loss = jnp.mean(discriminator_sample_outputs)
  loss = data_loss + samples_loss
  loss_components = {
      'disc_data_loss': data_loss,
      'disc_samples_loss': samples_loss}
  return LossOutput(loss, loss_components)


def wasserstein_generator_loss(discriminator_real_data_outputs,
                               discriminator_sample_outputs):
  """Wasserstein generator loss. See: https://arxiv.org/abs/1701.07875."""
  # adding discriminator op on real data (easier to interpret, but does not
  # affect the optimization of the generator.
  data_loss = jnp.mean(discriminator_real_data_outputs)
  samples_loss = jnp.mean(- discriminator_sample_outputs)
  loss = data_loss + samples_loss
  loss_components = {
      'gen_data_loss': data_loss,
      'gen_samples_loss': samples_loss}
  return LossOutput(loss, loss_components)


### ************* Penalties. *************


def wgan_gradient_penalty_loss(disc_transform, rng, real_data, samples):
  """The gradient penalty loss on an interpolation of data and samples.

  Proposed by https://arxiv.org/pdf/1704.00028.pdf.

  Args:
    disc_transform: Function which takes one argument, data, and returns the
      output of the discriminator on that data. It has to be a transformation of
       the discriminator haiku module (via functools.partial or a lambda) such
       that the discriminator parameters are used, in order to be able to
       take gradients.
    rng: A JAX PRNG.
    real_data: jnp.array, real data batch.
    samples: jnp.array, samples batch.

  Returns:
    A scalar, the loss due to the gradient penalty.
  """
  batch_size = real_data.shape[0]

  # Coefficient for the linear interpolation.
  # We need to compute the shape of the data so that we can pass in 2d codes
  # in case we used the DiscriminatedAutoencoderOptimizer, where we have a
  # code discriminator which operates on 2D data.
  alpha_shape = [batch_size] + [1] * (real_data.ndim - 1)
  alpha = jax.random.uniform(rng, shape=alpha_shape)
  interpolation = real_data + alpha * (samples - real_data)

  # Compute the gradients wrt to the output of the critic for each interpolated
  # example.
  fn = lambda x: jnp.squeeze(disc_transform(jnp.expand_dims(x, axis=0)))
  grads = jax.vmap(jax.grad(fn))(interpolation)
  assert grads.shape == interpolation.shape
  slopes = utils.batch_l2_norms(grads)
  assert slopes.shape == (batch_size,)

  return jnp.mean((slopes - 1)**2)

