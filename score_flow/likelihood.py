# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import jax
import flax
import jax.numpy as jnp
import numpy as np
import sde_lib
import logging
from scipy import integrate
from models import utils as mutils
from utils import get_div_fn, get_value_div_fn, batch_mul


def get_likelihood_fn(sde, model, inverse_scaler, hutchinson_type='Rademacher', rtol=1e-5, atol=1e-5, method='RK45',
                      eps=1e-5, use_score_mean=False):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """

  def drift_fn(state, x, t, score_mean_t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, 
                                    train=False, continuous=True,
                                    score_mean_t=score_mean_t)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  @jax.pmap
  def p_value_div_fn(state, x, t, eps, score_mean_t=None):
    """Pmapped divergence of the drift function."""
    value_div_fn = get_value_div_fn(lambda x, t: drift_fn(state, x, t, score_mean_t))
    return value_div_fn(x, t, eps)

  @jax.pmap
  def pmap_score_fn(step_rng, state, batch, vec_t):
    score_fn = mutils.get_score_fn(sde, model, state.params_ema, state.model_state, 
                                    train=False, continuous=True,
                                    score_mean_t=None)
    mean, std = sde.marginal_prob(batch, vec_t)
    perturbed_data = mean + std[:, None, None, None] * jax.random.normal(step_rng, batch.shape)
    return score_fn(perturbed_data, vec_t)[0]

  p_prior_logp_fn = jax.pmap(sde.prior_logp)  # Pmapped log-PDF of the SDE's prior distribution

  p_marginal_prob = jax.pmap(sde.marginal_prob)

  def likelihood_fn(prng, pstate, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      z: A JAX array of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    rng, step_rng = jax.random.split(flax.jax_utils.unreplicate(prng))
    shape = data.shape
    if hutchinson_type == 'Gaussian':
      epsilon = jax.random.normal(step_rng, shape)
    elif hutchinson_type == 'Rademacher':
      epsilon = jax.random.randint(step_rng, shape,
                                   minval=0, maxval=2).astype(jnp.float32) * 2 - 1
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def ode_func(t, x):
      sample = mutils.from_flattened_numpy(x[:-shape[0] * shape[1]], shape)
      vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t
      drift, logp_grad = p_value_div_fn(pstate, sample, vec_t, epsilon)
      drift = mutils.to_flattened_numpy(drift)
      logp_grad = mutils.to_flattened_numpy(logp_grad)
      return np.concatenate([drift, logp_grad], axis=0)

    init = jnp.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0)
    solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    t = solution.t
    zp = jnp.asarray(solution.y[:, -1])
    z = mutils.from_flattened_numpy(zp[:-shape[0] * shape[1]], shape)
    delta_logp = zp[-shape[0] * shape[1]:].reshape((shape[0], shape[1]))
    prior_logp = p_prior_logp_fn(z)

    bpd = -(prior_logp + delta_logp)

    N = np.prod(shape[2:])
    bpd = bpd / N / np.log(2.)

    # A hack to convert log-likelihoods to bits/dim
    # based on the gradient of the inverse data normalizer.
    offset = jnp.log2(jax.grad(inverse_scaler)(0.)) + 8.
    bpd += offset
    return bpd, z, t, nfe, solution

  def likelihood_fn_use_score_mean(prng, pstate, data, score_mean_dict=None, train_ds=None,
                                   scaler=None, inverse_scaler=None, n_estimates=1):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      z: A JAX array of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    logging.info("Get into function!!!")
    rng, step_rng = jax.random.split(flax.jax_utils.unreplicate(prng))
    shape = data.shape
    if hutchinson_type == 'Gaussian':
      epsilon = jax.random.normal(step_rng, shape)
    elif hutchinson_type == 'Rademacher':
      epsilon = jax.random.randint(step_rng, shape,
                                   minval=0, maxval=2).astype(jnp.float32) * 2 - 1
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def estimate_score_mean(rng, state, t):
      train_iter = iter(train_ds)
      score_sum, n_data, index = 0, 0, 0
      rng = flax.jax_utils.unreplicate(rng)
      while True:
        try:
          batch = next(train_iter)
        except StopIteration:
          break
        batch = jax.tree_map(lambda x: scaler(x._numpy()), batch)
        batch = batch['image'] # shape=1x1024x32x32x3
        vec_t = jnp.ones((batch.shape[0], batch.shape[1])) * t # shape=1x1024
        for _ in range(n_estimates):
          rng, step_rng = jax.random.split(rng)
          p_step_rng = flax.jax_utils.replicate(step_rng)
          score = pmap_score_fn(p_step_rng, state, batch, vec_t)
          score_sum += score.sum(0)
          n_data += n_estimates * batch.shape[0]
        index += 1
      return score_sum / n_data

    def ode_func(t, x):
      sample = mutils.from_flattened_numpy(x[:-shape[0] * shape[1]], shape) # shape=1x1024x32x32x3
      vec_t = jnp.ones((sample.shape[0], sample.shape[1])) * t # shape=1x1024

      t_str = "{:.4f}".format(t)

      if t_str in score_mean_dict:
        print('Already exists time ', t_str)
        score_mean_t = score_mean_dict[str(t_str)]
      else:
        print('Computing time ', t_str)
        score_mean_t = estimate_score_mean(prng, pstate, t)
        score_mean_dict[str(t_str)] = score_mean_t

      p_score_mean_t = flax.jax_utils.replicate(score_mean_t)
      drift, logp_grad = p_value_div_fn(pstate, sample, vec_t, epsilon, score_mean_t=p_score_mean_t)
      drift = mutils.to_flattened_numpy(drift)
      logp_grad = mutils.to_flattened_numpy(logp_grad)
      return np.concatenate([drift, logp_grad], axis=0)

    init = jnp.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0)
    solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
    nfe = solution.nfev
    t = solution.t
    zp = jnp.asarray(solution.y[:, -1])
    z = mutils.from_flattened_numpy(zp[:-shape[0] * shape[1]], shape)
    delta_logp = zp[-shape[0] * shape[1]:].reshape((shape[0], shape[1]))
    prior_logp = p_prior_logp_fn(z)

    bpd = -(prior_logp + delta_logp)

    N = np.prod(shape[2:])
    bpd = bpd / N / np.log(2.)

    # A hack to convert log-likelihoods to bits/dim
    # based on the gradient of the inverse data normalizer.
    offset = jnp.log2(jax.grad(inverse_scaler)(0.)) + 8.
    bpd += offset
    return bpd, score_mean_dict, z, t, nfe, solution

  return likelihood_fn if not use_score_mean else likelihood_fn_use_score_mean