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
"""Config for training GANs using Adam as a baseline."""

from jaxline import base_config
from ml_collections import config_dict


def get_config():
  """Return config object for training."""
  config = base_config.get_base_config()

  ## Experiment config.
  config.experiment_kwargs = config_dict.ConfigDict(
      dict(
          config=dict(
              random_seed=0,
              dataset='cifar10',
              data_processor='ImageProcessor',
              optimizers=dict(
                  discriminator=dict(
                      name='adam',
                      lr=1e-4,
                      kwargs=dict(
                          b1=0.5,
                          b2=0.999)),
                  generator=dict(
                      name='adam',
                      lr=1e-4,
                      kwargs=dict(
                          b1=0.5,
                          b2=0.999)),
                  ),
              nets=dict(  # See `nets.py`
                  discriminator='CifarDiscriminator',
                  disc_kwargs=dict(),
                  generator='CifarGenerator',
                  gen_kwargs=dict(),
              ),
              losses=dict(  # See `losses.py`
                  discriminator='discriminator_goodfellow_loss',
                  generator='generator_saturating_loss',
              ),
              penalties=dict(  # See `losses.py`
                  discriminator=None,
                  generator=None,
              ),
              param_transformers=dict(  # See `nets.py`
                  # discriminator='spectral_norm',
                  discriminator='no_op',
                  generator=None,
              ),
              training=dict(
                  simultaneous_updates=False,
                  runge_kutta_updates=False,
                  estimator_fn='unbiased_estimate_fn_lowest_variance',
                  # One of: disc_first, gen_first.
                  alternating_player_order='disc_first',
                  batch_size=128,
                  rk_disc_regularizer_weight_coeff=0.,
                  grad_regularizes=dict(
                      dd_coeffs_multiplier=dict(
                          disc=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=0.0,
                          ),
                          gen=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=0.0,
                          )),
                      explicit_non_dd_coeffs=dict(
                          disc=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=0.0,
                              ),
                          gen=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=0.0,
                          ))),
                  num_gen_updates=1,
                  num_disc_updates=1,
                  num_latents=128),
              eval=dict(
                  run_image_metrics=True,
                  batch_size=16,
                  # The number of data/sample splits to be used for evaluation.
                  num_eval_splits=5,
                  num_inception_images=10000),
          )))

  ## Training loop config.
  config.interval_type = 'steps'
  config.training_steps = int(6e5)
  config.log_tensors_interval = 100
  config.save_checkpoint_interval = 100
  config.train_checkpoint_all_hosts = False

  # Debugging info
  # Set the `init_ckpt_path` to a trained model for local training.
  # config.init_ckpt_path = ''
  # Change to evaluate a specific checkpoint
  config.restore_path = ''
  config.checkpoint_dir = '/tmp/dd_two_player_games'

  config.eval_specific_checkpoint_dir = ''
  # Change to False if you want to test checkpointing.
  config.delete_existing_local_checkpoints = True

  config.best_model_eval_metric = 'IS_mean'

  return config


