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
"""Config for training GANs using SGD (including explicit regularisation)."""

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
                      name='sgd',
                      clip=None,
                      lr=1e-2,
                      kwargs=dict(momentum=0.)),
                  generator=dict(
                      name='sgd',
                      clip=None,
                      lr=0.005,
                      kwargs=dict(momentum=0.)),
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
                  discriminator='spectral_norm',
                  generator=None,
              ),
              training=dict(
                  simultaneous_updates=True,
                  runge_kutta_updates=False,
                  # One of: disc_first, gen_first.
                  alternating_player_order='disc_first',
                  estimator_fn='unbiased_estimate_fn_lowest_variance',
                  # estimator_fn='biased_estimate_multiple_devices_grad_fn',
                  batch_size=128,
                  rk_disc_regularizer_weight_coeff=0.,
                  grad_regularizes=dict(
                      dd_coeffs_multiplier=dict(
                          disc=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=-1.0,
                          ),
                          gen=dict(
                              self_norm=0.0,
                              other_norm=0.0,
                              other_dot_prod=-1.0,
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
  config.training_steps = int(3e5)
  config.train_checkpoint_all_hosts = False
  config.log_train_data_interval = 10
  config.log_tensors_interval = 10
  config.save_checkpoint_interval = 100
  config.eval_specific_checkpoint_dir = ''
  config.restore_path = ''
  config.checkpoint_dir = '/tmp/dd_two_player_games'

  config.best_model_eval_metric = 'IS_mean'

  return config
