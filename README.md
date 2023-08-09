# Discretization Drift related code

Code for
["On a continuous time model of gradient descent dynamics and instability in deep learning"]
(https://arxiv.org/abs/2302.01952) can be found in the
`principal_flow_instability_single_objective/` directory.

## Discretization Drift in Two-Player Games

This is the code reproducing the ICML 2021 paper
["Discretization Drift in Two-Player Games"]
(https://arxiv.org/abs/2105.13922)
by Mihaela Rosca, Yan Wu, Benoit Dherin and
 David G.T. Barrett.

 The code uses JAX for training and TensorFlow (via TF-GAN) for evaluation.

 If you make use of any code in your work, please cite:

```
@article{rosca2021discretization,
  title={Discretization Drift in Two-Player Games},
  author={Rosca, Mihaela and Wu, Yan and Dherin, Benoit and Barrett, David GT},
  journal={International Conference on Machine Learning},
  year={2021}
}
```

## Running the code

You can run the code using the `run.sh` script:
```
./run.sh train
```

Note: you might have to edit the `run.sh` script to ensure the JAX installation
uses the right device you want to use (GPU, CPU, TPU) and the right drivers
(e.g. CUDA drivers for the GPU).


## Evaluation

Evaluation uses TF-GAN to compute the Inception Score / FID.

You can evaluate an existing checkpoint by overriding the `restore_path` field
in the config dict which was used for training, and specify the mode
as `eval` as input to `run.sh`:

You can run the code using the `run.sh` script:
```
./run.sh eval
```

## Code structure

The code is structured as follows:

  * `experiment.py`: the main file running the experiment. Contains the
      optimisation code (discriminator and generator updates) and glue code.
  * `gan.py`: The definition of the GAN module. Takes a discriminator and
      generator networks as well as loss functions and provides the
      discriminator and generator loss values for a set of inputs.
  * `gan_grads_calculator.py`: Module to compute gradients for the GAN modules
      (including using explicit regularisation).
  * `regularizer_estimates.py`: Computes gradients for explicit regularizers
      used both to cancel drift terms and in ODE-GAN.
  * `nets.py`: Networks used for the discriminator and generator.
  * `losses.py`: The definition of losses used for the discriminator and
      generator.
  * `drift_utils.py`: drift utilities.
  * `optim.py`: optimisation utilities.
  * `data_utils.py`: utilities regarding datasets.
  * `utils.py`: primarily tree manipulation utilities for JAX.
  * `model_utils.py`: A sampler used for evaluation.
  * Configuration file: all `*_config.py`.

## Example experiments

CIFAR-10:

  * SGD - this was used for all the experiments in the paper which include
    discretization drift, as well as the SGA and CO comparison. See
    `sgd_cifar_config.py`.
  * RungeKutta4 - used to compare against a method which has a lower order drift
    (5th order in learning rate). See `ode_gan_cifar_config.py`.
  * Adam - only used as a comparison with SGD. See `cifar_config.py`.

We also provide configurations for MNIST (`config.py`).

## Explicit regularisation

The coefficients for explicit regularisation can be specified in the
configuration files. The relevant fields are associated with the key
`grad_regularizes`:


```
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

```

The subfields are:

  * `dd_coeffs_multiplier`:  For each player, one can specify what coefficient
      to be used for the discretization drift terms: `self_norm`,
      the gradient norm of the player, `other_norm`, the gradient norm of the
      other player, `other_dot_prod`: the dot product between this player's loss
      wrt to the other player's parameters and the other player's loss and
      the other player's parameters.
      Note that the discretization drift terms include their coefficients
      (which are computed from learning rates). Thus, to cancel a drift term
      use `-1`. To strengthen the drift term proportional to the drift strength
      use `1`. To leave the drift as is (as defined by gradient descent), use 0.
  * `explicit_non_dd_coeffs`: Explicit regularization for each player. This can
      be used to add explicit regularization that does not depend on the drift
      coefficients.


Example usage of `grad_regularizes` to cancel the *interaction terms* of the
drift:

```
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

```

To cancel the interaction terms and strengthening the self terms:

```
grad_regularizes=dict(
    dd_coeffs_multiplier=dict(
        disc=dict(
            self_norm=1.0,
            other_norm=0.0,
            other_dot_prod=-1.0,
        ),
        gen=dict(
            self_norm=1.0,
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

```

For consensus optimisation with hyperparameter lambda:

```
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
            self_norm=lambda,
            other_norm=lambda,
            other_dot_prod=0.0,
            ),
        gen=dict(
            self_norm=lambda,
            other_norm=lambda,
            other_dot_prod=0.0,
        ))),

```

For SGA with hyperparameter lambda:

```
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
            other_norm=lambda,
            other_dot_prod=0.0,
            ),
        gen=dict(
            self_norm=0.0,
            other_norm=lambda,
            other_dot_prod=0.0,
        ))),

```


## Running tests

The code we provide comes with tests. To run the tests, simply run the
python test file from the virtual env created by the `run.sh`:

```
  source dd_venv/bin/activate
  PYTHONPATH=.::$PYTHONPATH python3 dd_two_player_games/regularizer_estimates_test.py
```

## Disclaimer

This is not an official Google product.
