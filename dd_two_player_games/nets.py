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
"""Networks to train GANs."""

import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def spectral_norm():
  # Ignore biases.
  return hk.SNParamsTree(ignore_regex='.*b$', eps=1e-12, n_steps=1)


def no_op():
  return NoOpTree()


class NoOpTree(hk.Module):
  """No op parameter transformer."""

  def __init__(self, track_singular_values=True):
    """Initializes an NoOpTree module."""
    super().__init__(name='NoOpTree')
    self._track_singular_values = track_singular_values

    if self._track_singular_values:
      self._sn_tree = spectral_norm()

  def __call__(self, tree, update_stats=True):
    if self._track_singular_values:
      # We do not use the return result, but want to update the states in case
      # we want to track the singular values without SN.
      self._sn_tree(tree, update_stats=update_stats)

    # No change to the input parameters.
    return tree


class ConvNet2DTranspose(hk.Module):
  """Conv Transposed Net. Does not activate last layer."""

  def __init__(self,
               output_channels,
               strides,
               kernel_shapes,
               use_batch_norm=False,
               batch_norm_config=None,
               name=None):
    super().__init__(name=name)

    if len(strides) == 1:
      strides = strides * len(output_channels)

    if len(output_channels) != len(strides):
      raise ValueError('The length of the output_channels and strides has'
                       'to be the same but was {} and {}'.format(
                           len(output_channels), len(strides)))

    if len(kernel_shapes) == 1:
      if len(kernel_shapes[0]) != 2:
        raise ValueError('Invalid kernel shapes {}'.format(kernel_shapes))
      kernel_shapes = kernel_shapes * len(output_channels)

    self._output_channels = output_channels
    self._strides = strides
    self._kernel_shapes = kernel_shapes
    self._num_layers = len(output_channels)

    self._use_batch_norm = use_batch_norm
    self._batch_norm_config = batch_norm_config

  def __call__(self, x, is_training=True, test_local_stats=False):
    for layer in range(self._num_layers):
      x = hk.Conv2DTranspose(
                    output_channels=self._output_channels[layer],
                    kernel_shape=self._kernel_shapes[layer],
                    stride=self._strides[layer],
                    padding='SAME')(x)
      if layer != self._num_layers - 1:
        if self._use_batch_norm:
          bn = hk.BatchNorm(**self._batch_norm_config)
          x = bn(x, is_training=is_training, test_local_stats=test_local_stats)

        x = jax.nn.relu(x)

    return x


class ConvNet2D(hk.Module):
  """Conv network."""

  def __init__(self,
               output_channels,
               strides,
               kernel_shapes,
               activation,
               name=None):
    super().__init__(name=name)

    if len(output_channels) != len(strides):
      raise ValueError('The length of the output_channels and strides has'
                       'to be the same but was {} and {}'.format(
                           len(output_channels), len(strides)))

    if len(kernel_shapes) == 1:
      if len(kernel_shapes[0]) != 2:
        raise ValueError('Invalid kernel shapes {}'.format(kernel_shapes))
      kernel_shapes = kernel_shapes * len(output_channels)

    self._output_channels = output_channels
    self._strides = strides

    self._kernel_shapes = kernel_shapes
    self._activation = activation
    self._num_layers = len(output_channels)

  def __call__(self, x, is_training=True):
    del is_training
    for layer in range(self._num_layers):
      x = hk.Conv2D(output_channels=self._output_channels[layer],
                    kernel_shape=self._kernel_shapes[layer],
                    stride=self._strides[layer],
                    padding='SAME')(x)
      if layer != self._num_layers -1:
        x = self._activation(x)
    return x


class CifarGenerator(hk.Module):
  """As in the SN paper."""

  def __init__(self, name='CifarGenerator'):
    super().__init__(name=name)

  def __call__(self, inputs, is_training=True, test_local_stats=False):
    batch_size = inputs.shape[0]
    first_shape = [4, 4, 512]
    up_tensor = hk.Linear(np.prod(first_shape))(inputs)
    first_tensor = jnp.reshape(up_tensor, [batch_size] + first_shape)

    net = ConvNet2DTranspose(
        output_channels=[256, 128, 64, 3],
        kernel_shapes=[(4, 4), (4, 4), (4, 4), (3, 3)],
        strides=[2, 2, 2, 1],
        use_batch_norm=True,
        batch_norm_config={
            'create_scale': True, 'create_offset': True, 'decay_rate': 0.999,
            'cross_replica_axis': 'i'})
    output = net(first_tensor, is_training=is_training,
                 test_local_stats=test_local_stats)
    return jnp.tanh(output)


class CifarDiscriminator(hk.Module):
  """Spectral normalization discriminator (metric) architecture."""

  def __init__(self):
    super().__init__(name='CifarDiscriminator')

  def __call__(self, inputs, is_training=True):
    activation = functools.partial(jax.nn.leaky_relu, negative_slope=0.1)
    net = ConvNet2D(
        output_channels=[64, 64, 128, 128, 256, 256, 512],
        kernel_shapes=[
            (3, 3), (4, 4), (3, 3), (4, 4), (3, 3), (4, 4), (3, 3)],
        strides=[1, 2, 1, 2, 1, 2, 1],
        activation=activation)

    output = hk.Linear(1)(
        hk.Flatten()(activation(net(inputs, is_training=is_training))))
    return output


class MnistGenerator(hk.Module):
  """MNIST Generator network."""

  def __init__(
      self, last_layer_activation=jnp.tanh, output_channels=(32, 1)):
    super().__init__(name='MnistGenerator')
    self._output_channels = output_channels
    self._last_layer_activation = last_layer_activation

  def __call__(self, x, is_training=True, test_local_stats=False):
    """Maps noise latents to images."""
    del is_training
    x = hk.Linear(7 * 7 * 64)(x)
    x = jnp.reshape(x, x.shape[:1] + (7, 7, 64))
    x = jax.nn.relu(x)

    x = ConvNet2DTranspose(
        output_channels=self._output_channels,
        kernel_shapes=[[5, 5]],
        strides=[2])(x)

    return self._last_layer_activation(x)


class MnistDiscriminator(hk.Module):
  """MNIST Discriminator network."""

  def __init__(self):
    super().__init__(name='MnistDiscriminator')

  def __call__(self, x, is_training=True):
    del is_training
    activation = functools.partial(jax.nn.leaky_relu, negative_slope=0.2)

    net = ConvNet2D(
        output_channels=[8, 16, 32, 64, 128],
        kernel_shapes=[[5, 5]],
        strides=[2, 1, 2, 1, 2],
        activation=activation)

    return hk.Linear(1)(hk.Flatten()(activation(net(x))))
