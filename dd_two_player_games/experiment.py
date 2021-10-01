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
"""Experiment file for GAN training."""


import collections
import datetime
import functools
import os
import signal
import threading

from absl import app
from absl import flags
from absl import logging

import dill
import jax
from jaxline import experiment
from jaxline import platform
from jaxline import utils as pipeline_utils
import numpy as np
import optax

from dd_two_player_games import data_utils
from dd_two_player_games import drift_utils
from dd_two_player_games import gan
from dd_two_player_games import gan_grads_calculator
from dd_two_player_games import losses
from dd_two_player_games import model_utils
from dd_two_player_games import nets
from dd_two_player_games import optim
from dd_two_player_games import regularizer_estimates
from dd_two_player_games import tfgan_inception_utils
from dd_two_player_games import utils


FLAGS = flags.FLAGS


UpdateDebugInfo = collections.namedtuple(
    'UpdateDebugInfo', ['log_dict', 'grads', 'update'])


def _make_coeff_tuple(config):
  disc_coeffs = drift_utils.PlayerRegularizationTerms(**config['disc'])
  gen_coeffs = drift_utils.PlayerRegularizationTerms(**config['gen'])
  return gan.GANTuple(disc=disc_coeffs, gen=gen_coeffs)


def get_explicit_coeffs(regularizer_config, dd_coeffs):
  """Obtain the coefficients used for explicit regularization."""
  mul = _make_coeff_tuple(regularizer_config.dd_coeffs_multiplier)
  dd_explicit = jax.tree_multimap(lambda x, y: x * y, mul, dd_coeffs)
  user_explicit = _make_coeff_tuple(regularizer_config.explicit_non_dd_coeffs)
  return jax.tree_multimap(lambda x, y: x + y, dd_explicit, user_explicit)


def sn_dict_for_logging(sn_dict):
  """Transform the spectral norm state for logging/display."""
  res = {}
  for k, v in sn_dict.items():
    sn_index = k.find('sn_params_tree')
    if sn_index > 0:
      k = k[sn_index:]

    res[k + '/sv'] = v['sigma']
  return res


def _check_config(config):
  if config.training.simultaneous_updates:
    if (config.training.num_disc_updates != 1 or
        config.training.num_gen_updates != 1):
      raise ValueError(
          'For simultaneous updates the number of updates per player has '
          'to be 1!')


def _get_data_processor(config):
  return getattr(data_utils, config.data_processor)


def _get_dataset(config, mode='train'):
  """Obtain dataset."""
  # Note: we always use the 'train' split of the data.
  # Currently mode only affects batch size.
  assert mode in ('train', 'eval')
  if mode == 'train':
    global_batch_size = config.training.batch_size
  else:
    global_batch_size = config.eval.batch_size

  num_devices = jax.device_count()
  logging.info(
      'Using %d devices, adjusting batch size accordingly!', num_devices)

  per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

  if ragged:
    raise ValueError(
        f'Global batch size {global_batch_size} must be divisible by '
        f'num devices {num_devices}')

  local_device_count = jax.local_device_count()
  logging.info('local_device_count: %d', local_device_count)

  if config.dataset in ('mnist', 'cifar10'):
    return data_utils.get_image_dataset(
        config.dataset, 'train',
        batch_dims=[local_device_count, per_device_batch_size],
        processor=_get_data_processor(config),
        shard_args=(jax.host_count(), jax.host_id()),
        seed=config.random_seed)
  else:
    raise NotImplementedError(
        'dataset {} not implemented'.format(config.dataset))


def _get_optim(player_optimizer_config):
  """Get optimizer (with learning rate scheduler)."""
  if 'scheduler' not in player_optimizer_config:
    optimizer = getattr(
        optax, player_optimizer_config.name)(
            player_optimizer_config.lr, **player_optimizer_config.kwargs)
    scheduler = optax.constant_schedule(player_optimizer_config.lr)
    return optimizer, scheduler

  scheduler_kwargs = dict(player_optimizer_config.scheduler_kwargs)
  scheduler_kwargs['boundaries_and_scales'] = {
      int(k): v for k, v in scheduler_kwargs['boundaries_and_scales'].items()}

  lr_schedule = getattr(optax, player_optimizer_config.scheduler)(
      init_value=player_optimizer_config.lr, **scheduler_kwargs)

  # Build the optimizers based on the learning rate scheduler.
  if 'sgd' == player_optimizer_config.name:
    clip = player_optimizer_config.clip
    if clip:
      optimizer = optax.chain(
          optax.scale_by_schedule(lr_schedule),
          optax.clip(clip),  # Clipping only for sgd.
          optax.scale(-1))
    else:
      optimizer = optax.chain(
          optax.scale_by_schedule(lr_schedule),
          optax.scale(-1))
  elif 'momentum' == player_optimizer_config.name:
    optimizer = optax.chain(
        optax.trace(decay=player_optimizer_config.kwargs.momentum,
                    nesterov=False),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1))
  elif 'adam' == player_optimizer_config.name:
    optimizer = optax.chain(
        optax.scale_by_adam(**player_optimizer_config.kwargs),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1))
  else:
    raise ValueError('Unsupported optimizer {}'.format(
        player_optimizer_config.name))

  return optimizer, lr_schedule


def _get_optimizers(config):
  """Construct optimizer from config."""
  disc_optimizer, disc_schedule = _get_optim(config.optimizers.discriminator)
  gen_optimizer, gen_schedule = _get_optim(config.optimizers.generator)
  optimizers = gan.GANTuple(disc=disc_optimizer, gen=gen_optimizer)
  schedules = gan.GANTuple(disc=disc_schedule, gen=gen_schedule)
  return optimizers, schedules


def _build_gan(config):
  """Build the GAN object."""
  players = gan.GANTuple(
      disc=getattr(nets, config.nets.discriminator),
      gen=getattr(nets, config.nets.generator))

  players_kwargs = gan.GANTuple(
      disc=config.nets.disc_kwargs, gen=config.nets.gen_kwargs)

  player_losses = gan.GANTuple(
      disc=getattr(losses, config.losses.discriminator),
      gen=getattr(losses, config.losses.generator))

  if config.penalties.discriminator:
    disc_penalty = gan.GANPenalty(
        fn=getattr(losses, config.penalties.discriminator[0]),
        coeff=config.penalties.discriminator[1])
  else:
    disc_penalty = None

  player_penalties = gan.GANTuple(disc=disc_penalty, gen=None)

  disc_transform = getattr(nets, config.param_transformers.discriminator)

  player_param_transformers = gan.GANTuple(disc=disc_transform, gen=None)

  return gan.GAN(
      players, player_losses, player_penalties, player_param_transformers,
      players_kwargs, config.training.num_latents)


class Experiment(experiment.AbstractExperiment):
  """GAN experiment."""

  CHECKPOINT_ATTRS = {
      '_params': 'params',
      '_opt_state': 'opt_state',
      '_state': 'state',
  }

  def __init__(self, mode, init_rng, config):

    super().__init__(mode=mode, init_rng=init_rng)

    # We control our own rngs, to avoid changes in jaxline that we do not
    # control.
    del init_rng
    init_rng = jax.random.PRNGKey(config.random_seed)

    self.mode = mode
    self.init_params_rng = pipeline_utils.bcast_local_devices(init_rng)

    # We need a different rng for each device since we want to obtain different
    # samples from each device.
    self.init_rng = jax.pmap(functools.partial(
        pipeline_utils.specialize_rng_host_device, axis_name='i',
        host_id=jax.host_id(),
        mode='unique_host_unique_device'), axis_name='i')(self.init_params_rng)

    _check_config(config)

    self._train_dataset = _get_dataset(config)
    self._eval_input = None
    self._gan = _build_gan(config)
    self._optimizers, self._lr_schedules = _get_optimizers(config)
    self._data_processor = _get_data_processor(config)
    self._simultaneous_updates = config.training.simultaneous_updates
    self._runge_kutta_updates = config.training.runge_kutta_updates

    self._num_updates = gan.GANTuple(
        disc=config.training.num_disc_updates,
        gen=config.training.num_gen_updates)
    self._learning_rates = gan.GANTuple(
        disc=config.optimizers.discriminator.lr,
        gen=config.optimizers.generator.lr)

    self._alternating_player_order = drift_utils.PlayerOrder[
        config.training.alternating_player_order]

    self._dd_coeffs = drift_utils.get_dd_coeffs(
        self._alternating_player_order,
        self._simultaneous_updates, self._learning_rates, self._num_updates)

    self._explicit_coeffs = get_explicit_coeffs(
        config.training.grad_regularizes, self._dd_coeffs)

    self._estimator_fn = getattr(
        regularizer_estimates, config.training.estimator_fn)

    self._gan_grads_calculator = gan_grads_calculator.GANGradsCalculator(
        self._gan, self._estimator_fn)

    if self._simultaneous_updates:
      if self._num_updates.disc != 1:
        raise ValueError('The number of discriminator updates in simultaneous '
                         'training must be 1!')
      if self._num_updates.gen != 1:
        raise ValueError('The number of generator updates in simultaneous '
                         'training must be 1!')

    # Always pmap the update steps.
    self._discriminator_update = jax.pmap(
        self._discriminator_update, axis_name='i',
        static_broadcasted_argnums=(5))

    self._generator_update = jax.pmap(
        self._generator_update, axis_name='i',
        static_broadcasted_argnums=(5))

    if self._runge_kutta_updates:
      self._runge_kutta = optim.RungeKutta(
          self._gan_grads_calculator.both_player_grads,
          config.training.runge_kutta_order)

      # When it comes to RK updates, we have the choice of whether
      # we want to add explicit regularization to each RK step (for this
      # we use self._explicit_coeffs) or not apply the penalty to each RK step
      # but add the explicit regularization gradients to the RK combined
      # gradients.
      # Note: while the explicit penalty is applied at the end of the RK step
      # it is still applied to the initial parameters, rather than those
      # obtained after the RK update has been performed.
      self._rk_after_explicit_coeffs = _make_coeff_tuple(
          config.training.grad_regularizes.rk_after_explicit_coeffs)
      self._runge_kutta_update = jax.pmap(
          self._runge_kutta_update, axis_name='i',
          static_broadcasted_argnums=(8, 9, 10))

    # Model parameters
    self._params = None
    self._opt_state = None
    self._state = None

    # eval state
    self.eval_config = config.eval
    self._run_image_metrics = config.eval.run_image_metrics
    self._image_metrics = None
    self._eval_dataset = _get_dataset(config, 'eval')

  def _maybe_init_training(self):
    data_batch = next(self._train_dataset)

    # If we are starting training now (not from a saved checkpoint), then
    # initialize parameters.
    if not self._params:
      logging.info('Initializing model parameters')

      init_gan = jax.pmap(self._gan.initial_params, axis_name='i')
      # Use the initial parameters rng to get the same initialization
      # between all replicas.
      self._params, self._state = init_gan(self.init_params_rng, data_batch)

      self._opt_state = gan.GANTuple(
          gen=jax.pmap(self._optimizers.gen.init, axis_name='i')(
              self._params.gen),
          disc=jax.pmap(self._optimizers.disc.init, axis_name='i')(
              self._params.disc))

  def new_rng(self, global_step):
    """Get a new rng.

    We fold in `global_step` so that we can split rngs inside a step without
    fearing of reusing the split value at a next step.

    Args:
      global_step: The global step.

    Returns:
      A new RNG - or pmapped array of rngs.
    """
    def split_rng(rng):
      return tuple(jax.random.split(jax.random.fold_in(rng, global_step[0])))
    self.init_rng, rng = jax.pmap(split_rng)(self.init_rng)
    return rng

  # Note: reads but does not modify the state of the object.
  # Returns the new discriminator parameters and optimizer state which
  # get updated in the `step` function.
  def _discriminator_update(
      self, data_batch, params, opt_state, state, rng_disc, is_training):
    disc_grads, disc_gan_loss_aux = self._gan_grads_calculator.disc_grads(
        params, state, data_batch, rng_disc, is_training, self._explicit_coeffs)
    disc_update, disc_opt_state = self._optimizers.disc.update(
        disc_grads, opt_state.disc)
    new_disc_params = optax.apply_updates(params.disc, disc_update)

    disc_debug_info = UpdateDebugInfo(
        log_dict=disc_gan_loss_aux.log_dict,
        grads=disc_grads, update=disc_update)
    state = disc_gan_loss_aux.state
    return new_disc_params, disc_opt_state, state, disc_debug_info

  # Note: reads but does not modify the state of the object.
  # Returns the new generator parameters and optimizer state which
  # get updated in the `step` function.
  def _generator_update(
      self, data_batch, params, opt_state, state, rng_gen, is_training):
    gen_grads, gen_gan_loss_aux = self._gan_grads_calculator.gen_grads(
        params, state, data_batch, rng_gen, is_training, self._explicit_coeffs)

    gen_update, gen_opt_state = self._optimizers.gen.update(
        gen_grads, opt_state.gen)
    new_gen_params = optax.apply_updates(params.gen, gen_update)

    gen_debug_info = UpdateDebugInfo(
        log_dict=gen_gan_loss_aux.log_dict, grads=gen_grads, update=gen_update)
    state = gen_gan_loss_aux.state
    return new_gen_params, gen_opt_state, state, gen_debug_info

  def discriminator_step(self, params, opt_state, state, global_step):
    for _ in range(self._num_updates.disc):
      data_batch = next(self._train_dataset)
      (disc_params, disc_opt_state,
       state, disc_update_info) = self._discriminator_update(
           data_batch, params, opt_state, state, self.new_rng(global_step),
           True)

      params = params._replace(disc=disc_params)
      opt_state = opt_state._replace(disc=disc_opt_state)
    return params, opt_state, state, disc_update_info

  def generator_step(self, params, opt_state, state, global_step):
    for _ in range(self._num_updates.gen):
      data_batch = next(self._train_dataset)
      (gen_params, gen_opt_state,
       state, gen_update_info) = self._generator_update(
           data_batch, params, opt_state, state, self.new_rng(global_step),
           True)

      params = params._replace(gen=gen_params)
      opt_state = opt_state._replace(gen=gen_opt_state)
    return params, opt_state, state, gen_update_info

  def _runge_kutta_update(self, params, opt_state, step_size,
                          states, data_batches, generator_batches,
                          disc_rngs, gen_rngs, is_training,
                          inside_rk_explicit_coeffs,
                          after_rk_explicit_coeffs):
    """Runge Kutta update."""
    rk_args = (states, data_batches, generator_batches,
               disc_rngs, gen_rngs, is_training, inside_rk_explicit_coeffs)
    gan_grads, gan_aux = self._runge_kutta.grad(params, step_size, *rk_args)

    disc_reg_grads, non_zero_coeff = self._gan_grads_calculator.disc_explicit_regularization_grads(
        params, states[0], data_batches[0],
        disc_rngs[0], is_training, after_rk_explicit_coeffs)

    disc_grads = utils.add_trees_with_coeff(
        acc=gan_grads.disc,
        mul=disc_reg_grads,
        coeff=non_zero_coeff)

    disc_grads = jax.lax.pmean(disc_grads, axis_name='i')

    gen_reg_grads, non_zero_coeff = self._gan_grads_calculator.gen_explicit_regularization_grads(
        params, states[0], generator_batches[0],
        gen_rngs[0], is_training, after_rk_explicit_coeffs)

    gen_grads = utils.add_trees_with_coeff(
        acc=gan_grads.gen,
        mul=gen_reg_grads,
        coeff=non_zero_coeff)
    gen_grads = jax.lax.pmean(gen_grads, axis_name='i')

    disc_update, disc_opt_state = self._optimizers.disc.update(
        disc_grads, opt_state.disc)
    disc_params = optax.apply_updates(params.disc, disc_update)

    gen_update, gen_opt_state = self._optimizers.gen.update(
        gen_grads, opt_state.gen)
    gen_params = optax.apply_updates(params.gen, gen_update)

    params = gan.GANTuple(disc=disc_params, gen=gen_params)
    opt_state = gan.GANTuple(disc=disc_opt_state, gen=gen_opt_state)

    return params, opt_state, gan_aux

  def _disc_lr(self, global_step):
    return jax.pmap(self._lr_schedules.disc)(global_step)

  def _runge_kutta_step(self, global_step):
    # We use the discriminator schedule for RK updates.
    order = self._runge_kutta.runge_kutta_order
    # CRUCIAL: these need to be tuples (especially the static arg is training)
    # otherwise jax recompiles every call and the code becomes extremely slow.
    states = tuple([self._state] * order)
    data_batches = tuple([next(self._train_dataset) for i in range(order)])
    generator_batches = tuple([next(self._train_dataset) for i in range(order)])
    disc_rngs = tuple([self.new_rng(global_step) for i in range(order)])
    gen_rngs = tuple([self.new_rng(global_step) for i in range(order)])
    is_training = tuple([True] * order)
    inside_rk_explicit_coeffs = tuple([self._explicit_coeffs] * order)
    # The coefficients after the RK step need no duplication since they
    # will be applied outside the RK optimizer.
    after_rk_explicit_coeffs = self._rk_after_explicit_coeffs
    params, opt_state, gan_aux = self._runge_kutta_update(
        self._params, self._opt_state, self._disc_lr(global_step),
        states, data_batches, generator_batches,
        disc_rngs, gen_rngs, is_training,
        inside_rk_explicit_coeffs,
        after_rk_explicit_coeffs)

    self._params = params
    self._opt_state = opt_state

    # Since we are doing simultaneous updates, we are updating the states
    # of the two players only after both updates.
    player_states = gan.GANTuple(
        disc=gan_aux.disc.state.players.disc,
        gen=gan_aux.gen.state.players.gen)
    param_transforms_states = gan.GANTuple(
        disc=gan_aux.disc.state.param_transforms.disc,
        gen=gan_aux.gen.state.param_transforms.gen)
    state = gan.GANState(players=player_states,
                         param_transforms=param_transforms_states)

    self._state = state
    logged_dict = {}
    logged_dict.update(gan_aux.disc.log_dict)
    logged_dict.update(gan_aux.gen.log_dict)
    return logged_dict

  def _simultaneous_updates_step(self, global_step):
    """Perform a simultaneous update step.

    Args:
      global_step: The global step of this update.

    Returns:
      A dictionary of logged values.

    Steps:
       * Discriminator step:
            * Compute discriminator gradients.
            * Apply discriminator gradients.
            * Update discriminator state.
      * Generator step:
            * Compute generator gradients using old discriminator parameters and
                old discriminator state.
            * Apply generator gradients.
            * Update generator state.
    """
    state = self._state
    opt_state = self._opt_state
    params = self._params

    # Discriminator.
    (params_from_disc_update, opt_state_from_disc_update,
     state_from_disc_update, disc_update_info) = self.discriminator_step(
         params, opt_state, state, global_step)

    # Generator.
    (params_from_gen_update, opt_state_from_gen_update,
     state_from_gen_update, gen_update_info) = self.generator_step(
         params, opt_state, state, global_step)

    def pick_gan_tuple(disc_tuple, gen_tuple):
      return gan.GANTuple(disc=disc_tuple.disc, gen=gen_tuple.gen)

    self._params = pick_gan_tuple(
        params_from_disc_update, params_from_gen_update)
    self._opt_state = pick_gan_tuple(
        opt_state_from_disc_update, opt_state_from_gen_update)

    # Since we are doing simultaneous updates, we are updating the states
    # of the two players only after both updates.
    player_states = pick_gan_tuple(
        state_from_disc_update.players, state_from_gen_update.players)
    param_transforms_states = pick_gan_tuple(
        state_from_disc_update.param_transforms,
        state_from_gen_update.param_transforms)

    state = gan.GANState(players=player_states,
                         param_transforms=param_transforms_states)

    self._state = state
    logged_dict = {}
    logged_dict.update(disc_update_info.log_dict)
    logged_dict.update(gen_update_info.log_dict)
    return logged_dict

  def _alternating_updates_step(self, global_step):
    """Perform an alternating updates step.

    Args:
      global_step: The global step of this update.

    Returns:
      A dictionary of logged values.

    Steps:
       * For the discriminator number of updates:
            * Compute discriminator gradients.
            * Apply discriminator gradients.
            * Update discriminator state.
      * For the generator number of updates:
            * Compute generator gradients using new discriminator parameters and
                new discriminator state.
            * Apply generator gradients.
            * Update generator state.
    """
    state = self._state
    opt_state = self._opt_state
    params = self._params

    logged_dict = {}

    if self._alternating_player_order == drift_utils.PlayerOrder.disc_first:
      step_fns = [self.discriminator_step, self.generator_step]
    else:
      step_fns = [self.generator_step, self.discriminator_step]

    for player_step_fn in step_fns:
      # Note: the generator update uses the new state obtained from the
      # discriminator update but does *NOT* update the discriminator state,
      # even though it does a discriminator forward pass.
      # This implies that if methods like Spectral Normalization are used,
      # the generator does a SN step so that it uses the normalized parameters,
      # but does not change the value of the power iteration vectors.
      # This is different compared to the Tensorflow implementation which used
      # custom getters.
      # Experiments show this makes no difference, and makes the theory more
      # consistent, since we assume that the generator does not change the
      # discriminator in its update.
      params, opt_state, state, player_update_info = player_step_fn(
          params, opt_state, state, global_step)

      logged_dict.update(player_update_info.log_dict)

    self._params = params
    self._opt_state = opt_state
    self._state = state

    return logged_dict

  def step(self, global_step, rng, **unused_kwargs):
    """Training step."""

    del rng  # We handle our own random number generation.

    logging.info('Training step %d', global_step)
    # Initialize parameters
    self._maybe_init_training()

    if self._simultaneous_updates:
      update_log_dict = self._simultaneous_updates_step(global_step)
    elif self._runge_kutta_updates:
      update_log_dict = self._runge_kutta_step(global_step)
    else:
      update_log_dict = self._alternating_updates_step(global_step)

    # We update the dictionary with information from the state.
    update_log_dict.update(
        sn_dict_for_logging(self._state.param_transforms.disc))

    # For losses, we average across devices - the same as averaging over
    # mini-batches.
    # For SN singular values:
    # Since we are doing sync training all SN values should be the same, but
    # we average for numerical stability.
    update_log_dict = utils.average_dict_values(update_log_dict)

    return update_log_dict

  def evaluate(self, global_step, rng, writer, **unused_args):
    """Evaluation step."""
    del rng  # We handle our own random number generation.

    sampler = model_utils.Sampler(self._gan, self._params, self._state)
    samples = sampler.sample_batch(
        self.eval_config.batch_size, self.new_rng(global_step))
    data_batch = next(self._eval_dataset)

    if self._run_image_metrics:
      if not self._image_metrics:
        # No dataset hash, recompute dataset metrics.
        self._image_metrics = tfgan_inception_utils.InceptionMetrics(
            self._eval_dataset,
            num_splits=self.eval_config.num_eval_splits)

      eval_metrics = self._image_metrics.get_metrics(
          sampler, self.eval_config.num_inception_images,
          self.eval_config.batch_size, rng=self.new_rng(global_step),
          no_pmap=False, sample_postprocess_fn=None)
      eval_metrics = {k: float(v) for k, v in eval_metrics.items()}
    else:
      # Add dummy metrics so that jaxline does not complain that we do
      # not provide the metrics required for the best checkpoint.
      eval_metrics = {'IS_mean': 0, 'FID': 0}

    logging.info(eval_metrics)

    if writer:
      global_step = np.array(pipeline_utils.get_first(global_step))

      def process_for_display(images):
        images = self._data_processor.postprocess(images)
        assert len(images.shape) == 5
        return np.concatenate(images, axis=0)

      display_samples = process_for_display(samples)
      display_data = process_for_display(data_batch)

      writer.write_images(
          global_step,
          {'samples': display_samples,
           'data': display_data,
           'xd_samples': display_samples})

    eval_metrics = utils.average_dict_values(eval_metrics)
    return eval_metrics


def safe_dict_update(dict1, dict2):
  for key, value in dict2.items():
    if key in dict1:
      raise ValueError('key already present {}'.format(key))
    dict1[key] = value


def _restore_state_to_in_memory_checkpointer(restore_path):
  """Initializes experiment state from a checkpoint."""

  # Load pretrained experiment state.
  python_state_path = os.path.join(restore_path, 'checkpoint.dill')
  with open(python_state_path, 'rb') as f:
    pretrained_state = dill.load(f)
  logging.info('Restored checkpoint from %s', python_state_path)

  # Assign state to a dummy experiment instance for the in-memory checkpointer,
  # broadcasting to devices.
  dummy_experiment = Experiment(
      mode='train', init_rng=0, config=FLAGS.config.experiment_kwargs.config)
  for attribute, key in Experiment.CHECKPOINT_ATTRS.items():
    setattr(dummy_experiment, attribute,
            pipeline_utils.bcast_local_devices(pretrained_state[key]))

  jaxline_state = dict(
      global_step=pretrained_state['global_step'],
      experiment_module=dummy_experiment)
  snapshot = pipeline_utils.SnapshotNT(0, jaxline_state)

  # Finally, seed the jaxline `pipeline_utils.InMemoryCheckpointer` global dict.
  pipeline_utils.GLOBAL_CHECKPOINT_DICT['latest'] = pipeline_utils.CheckpointNT(
      threading.local(), [snapshot])


def _get_step_date_label(global_step):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split('.')[0]
  return f'step_{global_step}_{date_str}'


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info('Saving model.')
  for (checkpoint_name,
       checkpoint) in pipeline_utils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    pickle_nest = checkpoint.history[-1].pickle_nest
    global_step = pickle_nest['global_step']

    state_dict = {'global_step': global_step}
    for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
      state_dict[key] = pipeline_utils.get_first(
          getattr(pickle_nest['experiment_module'], attribute))
    save_dir = os.path.join(
        save_path, checkpoint_name, _get_step_date_label(global_step))
    python_state_path = os.path.join(save_dir, 'checkpoint.dill')
    os.makedirs(save_dir, exist_ok=True)
    with open(python_state_path, 'wb') as f:
      dill.dump(state_dict, f)
    logging.info(
        'Saved "%s" checkpoint to %s', checkpoint_name, python_state_path)


def _setup_signals(save_model_fn):
  """Sets up a signal for model saving."""
  # Save a model on Ctrl+C.
  def sigint_handler(unused_sig, unused_frame):
    # Ideally, rather than saving immediately, we would then "wait" for a good
    # time to save. In practice this reads from an in-memory checkpoint that
    # only saves every 30 seconds or so, so chances of race conditions are very
    # small.
    save_model_fn()
    logging.info(r'Use `Ctrl+\` to save and exit.')

  # Exit on `Ctrl+\`, saving a model.
  prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)
  def sigquit_handler(unused_sig, unused_frame):
    # Restore previous handler early, just in case something goes wrong in the
    # next lines, so it is possible to press again and exit.
    signal.signal(signal.SIGQUIT, prev_sigquit_handler)
    save_model_fn()
    logging.info(r'Exiting on `Ctrl+\`')

    # Re-raise for clean exit.
    os.kill(os.getpid(), signal.SIGQUIT)

  signal.signal(signal.SIGINT, sigint_handler)
  signal.signal(signal.SIGQUIT, sigquit_handler)


def main(argv, experiment_class: experiment.AbstractExperiment):

  # Maybe restore a model.
  restore_path = FLAGS.config.restore_path
  if restore_path:
    _restore_state_to_in_memory_checkpointer(restore_path)

  # Maybe save a model.
  save_dir = os.path.join(FLAGS.config.checkpoint_dir, 'models')
  if FLAGS.config.one_off_evaluate:
    save_model_fn = lambda: None  # No need to save checkpoint in this case.
  else:
    save_model_fn = functools.partial(
        _save_state_from_in_memory_checkpointer, save_dir, experiment_class)
  _setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

  try:
    platform.main(experiment_class, argv)
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.


if __name__ == '__main__':
  flags.mark_flag_as_required('config')
  app.run(lambda argv: main(argv, Experiment))  # pytype: disable=wrong-arg-types
