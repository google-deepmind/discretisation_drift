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
"""Utilities for the drift."""


import collections
import enum
import jax

from dd_two_player_games import gan


class PlayerOrder(enum.Enum):
  disc_first = 0
  gen_first = 1


PlayerRegularizationTerms = collections.namedtuple(
    'PlayerRegularizationTerms', [
        'self_norm',  # IDD
        'other_dot_prod',  # IDD
        'other_norm',  # Used for other regularizers, such as ODE-GAN.

    ]
)


def get_dd_coeffs(
    alternating_player_order, simultaneous_updates,
    learning_rates, num_updates):
  """Obtain the implicit regularization coefficients."""
  if simultaneous_updates:
    def coeffs(lr):
      return PlayerRegularizationTerms(
          self_norm=1./2 * lr,
          other_norm=0,
          other_dot_prod=1./2 * lr)
    return jax.tree_map(coeffs, learning_rates)
  else:
    if alternating_player_order == PlayerOrder.disc_first:
      first_player = 'disc'
      second_player = 'gen'
    else:
      first_player = 'gen'
      second_player = 'disc'

    first_player_lr = getattr(learning_rates, first_player)
    second_player_lr = getattr(learning_rates, second_player)
    first_player_coeffs = PlayerRegularizationTerms(
        self_norm=1./(2 * getattr(num_updates, first_player)) * first_player_lr,
        other_norm=0,
        other_dot_prod=1./2 * first_player_lr)

    lr_ratio = first_player_lr / second_player_lr
    second_player_coeffs = PlayerRegularizationTerms(
        self_norm=1./(2 * getattr(
            num_updates, second_player)) *  second_player_lr,
        other_norm=0,
        other_dot_prod=-1./2 * second_player_lr *(2 * lr_ratio - 1))
    return gan.GANTuple(** {first_player: first_player_coeffs,
                            second_player: second_player_coeffs})


