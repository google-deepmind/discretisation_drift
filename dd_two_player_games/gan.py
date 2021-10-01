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
"""Code to glue together the two GAN players."""
import collections
import functools

import haiku as hk
import immutabledict
import jax
import jax.numpy as jnp


GANTuple = collections.namedtuple('GANTuple', ['disc', 'gen'])
GANPenalty = collections.namedtuple('GANPenalty', ['fn', 'coeff'])
GANState = collections.namedtuple('GANState', ['players', 'param_transforms'])
GANLossAux = collections.namedtuple(
    'GANLossAux', ['state', 'log_dict', 'loss_components'])


def get_player_fn(player, player_kwargs):
  def player_fn(*args, **kwargs):
    return player(**player_kwargs)(*args, **kwargs)
  return player_fn


def filter_out_new_training_state(new_state, old_state):
  """Keeps only the training state from the old state.

  The training state is defined as anything that will be used
  during training time. All the state used at training time will be taken
  from `old_state`, while states needed for evaluation (such as batch norm
  statistics), will be kept from `new_state`.

  Filtering states this way allows us to have the best evaluation statistics
  while not leaking updates between the two players.

  Args:
    new_state: Dict composing of the new states (module_name to state).
    old_state: Dict composing of the old states (module_name to state).

  Returns:
    The resulting state after choosing components from old_state and new_state
    based on their interaction with training.
  """
  assert new_state.keys() == old_state.keys()
  out = collections.defaultdict(dict)

  for module_name in new_state.keys():
    if 'batch_norm' in module_name:
      out[module_name] = new_state[module_name]
    else:
      out[module_name] = old_state[module_name]

  return hk.data_structures.to_haiku_dict(out)


class GAN:
  """A module which applies the relevant losses to the disc and generator."""

  def __init__(
      self, players_hk, losses, penalties,
      player_param_transformers, players_kwargs, num_latents):
    """Initialises the module.

    Args:
      players_hk: A GANTuple containing the haiku module constructors to be
        passed to `hk.transform_with_state`. The discriminator is assumed to
        produce the right output which can then be passed into the corresponding
        losses.
      losses: A GANTuple containing the losses used to train the model.
        The current API assumes that losses both for the discriminator and
        generator are of the form:
          ```def loss(discriminator_real_output, discriminator_sample_output):
            ...
            return loss
          ```

          where `discriminator_real_output`, `discriminator_sample_output` are
            the discriminator output on real data and fake data respectively.

          The loss returned is a jax scalar.

      penalties: A GANTuple containing optional GANPenalty that can be used
        for training. If no penalty should be used for one of the players, pass
        None.
      player_param_transformers: A GANTuple containing parameter transformers
        that are applied before a forward pass of the players.
      players_kwargs: A GANTuple containing the keyword args to build the two
        players.
      num_latents: Integer, the number of latents used.
    """
    # Define the Haiku network transforms.
    gen_transform = hk.without_apply_rng(
        hk.transform_with_state(
            get_player_fn(players_hk.gen, players_kwargs.gen)))
    disc_transform = hk.without_apply_rng(
        hk.transform_with_state(
            get_player_fn(players_hk.disc, players_kwargs.disc)))

    if player_param_transformers.disc:
      self._disc_param_transform = hk.without_apply_rng(
          hk.transform_with_state(
              get_player_fn(player_param_transformers.disc, {})))
    else:
      self._disc_param_transform = None

    self._transforms = GANTuple(disc=disc_transform, gen=gen_transform)
    self._penalties = penalties
    self._losses = losses
    self._num_latents = num_latents

  def initial_params(self, rng, batch):
    """Returns the initial parameters."""
    # Generate dummy latents for the generator.
    dummy_latents = jnp.zeros((batch.shape[0], self._num_latents))

    # Get initial network parameters.
    rng_gen, rng_disc, rng_disc_transform = jax.random.split(rng, 3)

    gen_init_params, gen_init_state = self._transforms.gen.init(
        rng_gen, dummy_latents)
    disc_init_params, disc_init_state = self._transforms.disc.init(
        rng_disc, batch)

    init_params = GANTuple(gen=gen_init_params, disc=disc_init_params)
    init_players_state = GANTuple(gen=gen_init_state, disc=disc_init_state)

    if self._disc_param_transform:
      _, init_disc_params_transform_state = self._disc_param_transform.init(
          rng_disc_transform, init_params.disc)
    else:
      init_disc_params_transform_state = None

    init_param_transforms_states = GANTuple(
        disc=init_disc_params_transform_state, gen=None)
    init_state = GANState(
        players=init_players_state,
        param_transforms=init_param_transforms_states)

    return init_params, init_state

  @functools.partial(jax.jit, static_argnums=(0, 4, 5))
  def sample(self, gen_params, gen_state, rng, num_samples, gen_kwargs):
    """Generates images from noise latents."""
    latents = jax.random.normal(rng, shape=(num_samples, self._num_latents))
    return self._transforms.gen.apply(
        gen_params, gen_state, latents, **gen_kwargs)

  @functools.partial(jax.jit, static_argnums=(0, 4, 5))
  def disc_forward(self, disc_params, state, disc_inputs, is_training,
                   update_stats):
    """Discriminator forward pass with optional parameter transform."""
    if self._disc_param_transform:
      disc_params, disc_params_transform_state = self._disc_param_transform.apply(
          {}, state.param_transforms.disc, disc_params,
          update_stats=update_stats)
    else:
      disc_params_transform_state = ()

    if self._disc_param_transform:
      disc_params, disc_params_transform_state = self._disc_param_transform.apply(
          {}, state.param_transforms.disc, disc_params,
          update_stats=update_stats)
    else:
      disc_params_transform_state = ()

    disc_outputs, disc_state = self._transforms.disc.apply(
        disc_params, state.players.disc, disc_inputs, is_training)

    return disc_outputs, disc_state, disc_params, disc_params_transform_state

  @functools.partial(jax.jit, static_argnums=(0, 5))
  def disc_loss(self, params, state, data_batch, rng, is_training):
    """Compute discriminator loss. Only updates the discriminator state."""
    num_data = data_batch.shape[0]

    gen_kwargs = immutabledict.immutabledict(
        collections.OrderedDict([('is_training', is_training)]))
    samples, gen_state = self.sample(
        params.gen, state.players.gen, rng, num_data, gen_kwargs)
    # Update the part of the state which is used for evaluation, not training.
    # During training we do not want to have the players update each others'
    # state.
    # For example, we can update BatchNorm state since this is not used in
    # training, but we do not want to update SpectralNormalization state.
    gen_state = filter_out_new_training_state(gen_state, state.players.gen)

    # We merge the inputs to the discriminator and do it one pass in order
    # to ensure that we update the discriminator state only once and that
    # the real data and fake data obtain the same state.
    disc_inputs = jnp.concatenate((data_batch, samples), axis=0)

    (disc_outputs, disc_state,
     new_disc_params, disc_params_transform_state) = self.disc_forward(
         params.disc, state, disc_inputs, is_training, is_training)

    data_disc_output, samples_disc_output = jnp.split(
        disc_outputs, [data_batch.shape[0],], axis=0)

    loss, loss_components = self._losses.disc(
        data_disc_output, samples_disc_output)
    if self._penalties.disc:
      penalty_fn = self._penalties.disc.fn
      penalty_coeff = self._penalties.disc.coeff
      def disc_apply(x):
        # Note: we do not update the state from the penalty run.
        return self._transforms.disc.apply(
            new_disc_params, disc_state, x, is_training)[0]
      loss += penalty_coeff * penalty_fn(disc_apply, rng, data_batch, samples)

    player_states = GANTuple(gen=gen_state, disc=disc_state)
    param_transforms_states = state.param_transforms._replace(
        disc=disc_params_transform_state)
    state = GANState(players=player_states,
                     param_transforms=param_transforms_states)

    logged_dict = {'disc_loss': loss}
    aux = GANLossAux(
        state=state, log_dict=logged_dict, loss_components=loss_components)
    return loss, aux

  @functools.partial(jax.jit, static_argnums=(0, 5))
  def gen_loss(self, params, state, data_batch, rng, is_training):
    """Compute generator loss.  Only updates generator state."""
    num_data = data_batch.shape[0]
    gen_kwargs = immutabledict.immutabledict(
        collections.OrderedDict([('is_training', is_training)]))
    samples, gen_state = self.sample(
        params.gen, state.players.gen, rng, num_data, gen_kwargs)

    # We merge the inputs to the discriminator and do it one pass in order
    # to ensure that we update the discriminator state only once and that
    # the real data and fake data obtain the same state.
    disc_inputs = jnp.concatenate((data_batch, samples), axis=0)

    # Set is_training to the given values for the discriminator in case it has
    # state that changes during training. We pass update_stats to be False since
    # we do not want to update the discriminator parameter transformers in
    # the generator forward pass.
    (disc_outputs, disc_state, _, _) = self.disc_forward(
        params.disc, state, disc_inputs, is_training, False)

    data_disc_output, samples_disc_output = jnp.split(
        disc_outputs, [data_batch.shape[0],], axis=0)

    # Update the part of the state which is used for evaluation, not training.
    # During training we do not want to have the players update each others'
    # state.
    # For example, we can update BatchNorm state since this is not used in
    # training, but we do not want to update SpectralNormalization state.
    disc_state = filter_out_new_training_state(disc_state, state.players.disc)
    player_states = GANTuple(gen=gen_state, disc=disc_state)
    # No change to param transforms.
    state = state._replace(players=player_states)
    loss, loss_components = self._losses.gen(
        data_disc_output, samples_disc_output)

    logged_dict = {'gen_loss': loss}
    aux = GANLossAux(
        state=state, log_dict=logged_dict, loss_components=loss_components)
    return loss, aux

  @property
  def disc_loss_grad_fn(self):
    return jax.grad(lambda *args: self.disc_loss(*args)[0])

  @property
  def gen_loss_grad_fn(self):
    return jax.grad(lambda *args: self.gen_loss(*args)[0])

  def disc_loss_fn_disc_grads(self, *args):
    """Computes discriminator loss gradients wrt to discriminator params."""
    return self.disc_loss_grad_fn(*args).disc

  def disc_loss_fn_gen_grads(self, *args):
    """Computes discriminator loss gradients wrt to generator params."""
    return self.disc_loss_grad_fn(*args).gen

  def gen_loss_fn_disc_grads(self, *args):
    """Computes generator loss gradients wrt to discriminator params."""
    return self.gen_loss_grad_fn(*args).disc

  def gen_loss_fn_gen_grads(self, *args):
    """Computes loss loss gradients wrt to generator params."""
    return self.gen_loss_grad_fn(*args).gen

  @property
  def disc_loss_components(self):
    return lambda *args: self.disc_loss(*args)[1].loss_components

  @property
  def gen_loss_components(self):
    return lambda *args: self.gen_loss(*args)[1].loss_components
