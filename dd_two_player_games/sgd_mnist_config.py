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
"""Config for training GANs on MNIST with SGD."""

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
              dataset='mnist',
              num_eval_samples=100,
              data_processor='ImageProcessor',
              optimizers=dict(
                  discriminator=dict(
                      name='sgd',
                      kwargs=dict(momentum=0.),
                      lr=1e-2),
                  generator=dict(
                      name='sgd',
                      kwargs=dict(momentum=0.),
                      lr=1e-2),
                  ),
              nets=dict(  # See `nets.py`
                  discriminator='MnistDiscriminator',
                  generator='MnistGenerator',
                  disc_kwargs=dict(),
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
                  discriminator='no_op',
                  generator=None,
              ),
              training=dict(
                  simultaneous_updates=True,
                  runge_kutta_updates=False,
                  estimator_fn='biased_estimate_grad_fn',
                  batch_size=64,
                  rk_disc_regularizer_weight_coeff=0.,
                  grad_regularizes=dict(
                      dd_coeffs_multiplier=dict(
                          disc_reg_disc_norm=0.0,
                          disc_reg_gen_dot_prod=0.0,
                          gen_reg_disc_dot_prod=0.0,
                          gen_reg_gen_norm=0.0,
                          gen_reg_disc_norm=0.0,
                          disc_reg_gen_norm=0.0,
                      ),
                      explicit_non_dd_coeffs=dict(
                          disc_reg_disc_norm=0.0,
                          disc_reg_gen_dot_prod=0.0,
                          gen_reg_disc_dot_prod=0.0,
                          disc_reg_gen_norm=0.0,
                          gen_reg_disc_norm=0.0,
                          gen_reg_gen_norm=0.0)),
                  num_gen_updates=1,
                  num_disc_updates=1,
                  num_latents=128),
              eval=dict(
                  run_image_metrics=False,
                  batch_size=16,
                  # The number of data/sample splits to be used for evaluation.
                  num_eval_splits=5,
                  num_inception_images=10000),
          )))

  ## Training loop config.
  config.training_steps = int(1e5)
  config.log_train_data_interval = 60
  config.log_tensors_interval = 60
  config.save_checkpoint_interval = 300
  config.eval_specific_checkpoint_dir = ''

  return config


