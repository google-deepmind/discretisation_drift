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
"""Drift utils test."""


from absl.testing import absltest
from absl.testing import parameterized


from dd_two_player_games import drift_utils
from dd_two_player_games import gan


LEARNING_RATE_TUPLES = [
    (0.01, 0.01),
    (0.01, 0.05),
    (0.05, 0.01),
    (0.0001, 0.5)]


class DriftUtilsTest(parameterized.TestCase):
  """Test class to ensure drift coefficients are computed correctly.

  Ensures that the drift coefficients in two-player games are
  computed as for the math for:
    * simultaneous updates.
    * alternating updates (for both player orders).
  """

  @parameterized.parameters(LEARNING_RATE_TUPLES)
  def test_sim_updates(self, disc_lr, gen_lr):
    # player order does not matter.
    # the number of updates does not matter for simultaneous updates.
    learning_rates = gan.GANTuple(disc=disc_lr, gen=gen_lr)
    drift_coeffs = drift_utils.get_dd_coeffs(
        None, True, learning_rates, num_updates=None)

    self.assertEqual(drift_coeffs.disc.self_norm, 0.5 * disc_lr)
    self.assertEqual(drift_coeffs.disc.other_norm, 0.0)
    self.assertEqual(drift_coeffs.disc.other_dot_prod, 0.5 * disc_lr)

    self.assertEqual(drift_coeffs.gen.self_norm, 0.5 * gen_lr)
    self.assertEqual(drift_coeffs.gen.other_norm, 0.0)
    self.assertEqual(drift_coeffs.gen.other_dot_prod, 0.5 * gen_lr)

  @parameterized.parameters(LEARNING_RATE_TUPLES)
  def test_alt_updates(self, disc_lr, gen_lr):
    learning_rates = gan.GANTuple(disc=disc_lr, gen=gen_lr)
    num_updates = gan.GANTuple(disc=1, gen=1)
    drift_coeffs = drift_utils.get_dd_coeffs(
        drift_utils.PlayerOrder.disc_first, False, learning_rates,
        num_updates=num_updates)

    self.assertEqual(drift_coeffs.disc.self_norm, 0.5 * disc_lr)
    self.assertEqual(drift_coeffs.disc.other_norm, 0.0)
    self.assertEqual(drift_coeffs.disc.other_dot_prod, 0.5 * disc_lr)

    self.assertEqual(drift_coeffs.gen.self_norm, 0.5 * gen_lr)
    self.assertEqual(drift_coeffs.gen.other_norm, 0.0)
    self.assertEqual(
        drift_coeffs.gen.other_dot_prod,
        0.5 * gen_lr * (1 - 2 * disc_lr / gen_lr))

  @parameterized.parameters(LEARNING_RATE_TUPLES)
  def test_alt_updates_change_player_order(self, disc_lr, gen_lr):
    learning_rates = gan.GANTuple(disc=disc_lr, gen=gen_lr)
    num_updates = gan.GANTuple(disc=1, gen=1)
    drift_coeffs = drift_utils.get_dd_coeffs(
        drift_utils.PlayerOrder.gen_first, False, learning_rates,
        num_updates=num_updates)

    self.assertEqual(drift_coeffs.disc.self_norm, 0.5 * disc_lr)
    self.assertEqual(drift_coeffs.disc.other_norm, 0.0)
    self.assertEqual(
        drift_coeffs.disc.other_dot_prod,
        0.5 * disc_lr * (1 - 2 * gen_lr / disc_lr))

    self.assertEqual(drift_coeffs.gen.self_norm, 0.5 * gen_lr)
    self.assertEqual(drift_coeffs.gen.other_norm, 0.0)
    self.assertEqual(drift_coeffs.gen.other_dot_prod, 0.5 * gen_lr)


if __name__ == '__main__':
  absltest.main()
