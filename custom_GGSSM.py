from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import necessary functions from dynamax
from dynamax.generalized_gaussian_ssm import ParamsGGSSM, GeneralizedGaussianSSM, EKFIntegrals
from dynamax.generalized_gaussian_ssm import conditional_moments_gaussian_smoother
from dynamax.linear_gaussian_ssm import LinearGaussianSSM


# Helper function to create a rotating linear system
def random_rotation(dim, key=0, theta=None):
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    key1, key2 = jr.split(key)

    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * jnp.pi * jr.uniform(key1)

    if dim == 1:
        return jr.uniform(key1) * jnp.eye(1)

    rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    out = jnp.eye(dim)
    out = out.at[:2, :2].set(rot)
    q = jnp.linalg.qr(jr.uniform(key2, shape=(dim, dim)))[0]
    return q.dot(out).dot(q.T)


class ExtendedGeneralizedGaussianSSM(GeneralizedGaussianSSM):
    def initialize(self, key):
        initial_mean = jnp.zeros(self.state_dim)
        initial_covariance = jnp.eye(self.state_dim)
        dynamics_matrix = random_rotation(self.state_dim, key)
        dynamics_covariance = jnp.eye(self.state_dim)
        emission_matrix = jr.normal(key, shape=(self.emission_dim, self.state_dim))
        emission_covariance = jnp.eye(self.emission_dim)

        return {'GGSSM' : ParamsGGSSM(
            initial_mean = initial_mean,
            initial_covariance = initial_covariance,
            dynamics_function = lambda z: dynamics_matrix @ z,
            dynamics_covariance = dynamics_covariance,
            emission_mean_function = lambda z: jnp.exp(poisson_weights @ z),
            emission_cov_function = lambda z: jnp.diag(jnp.exp(poisson_weights @ z)),
            emission_dist = lambda mu, Sigma: Pois(log_rate = jnp.log(mu))
        )}


def sample_poisson(model, params, num_steps, num_trials, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    def _sample(key):
        states, emissions = model.sample(params, num_timesteps=num_steps, key=key)
        return states, emissions

    if num_trials > 1:
        batch_keys = jr.split(key, num_trials)
        states, emissions = vmap(_sample)(batch_keys)
    else:
        states, emissions = _sample(key)

    return states, emissions


def sample_linear(model, params, num_steps, num_trials, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    def _sample(key):
        states, emissions = model.sample(params, num_timesteps=num_steps, key=key)
        return states, emissions

    if num_trials > 1:
        batch_keys = jr.split(key, num_trials)
        states, emissions = vmap(_sample)(batch_keys)
    else:
        states, emissions = _sample(key)

    return states, emissions


def main():
    key = jr.PRNGKey(0)
    poisson_model = ExtendedGeneralizedGaussianSSM(state_dim=2, emission_dim=5)
    params_ggssm = poisson_model.initialize(key)

    linear_model = LinearGaussianSSM(state_dim=2, emission_dim=5)
    params_lgssm, _ = linear_model.initialize(key)

    num_steps, num_trials = 200, 3
    all_states_ggssm, all_emissions_ggssm = sample_poisson(poisson_model, params_ggssm, num_steps, num_trials)
    all_states_lgssm, all_emissions_lgssm = sample_linear(linear_model, params_lgssm, num_steps, num_trials)

    # Plotting results for comparison
    fig = plt.figure(constrained_layout=True, figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Plot Poisson GGSSM emissions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(all_emissions_ggssm[0])
    ax1.set_title('Poisson GGSSM Emissions (Trial 1)')

    # Plot Poisson GGSSM states
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(all_states_ggssm[0])
    ax2.set_title('Poisson GGSSM States (Trial 1)')

    # Plot Linear Gaussian SSM emissions
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(all_emissions_lgssm[0])
    ax3.set_title('Linear Gaussian SSM Emissions (Trial 1)')

    # Plot Linear Gaussian SSM states
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(all_states_lgssm[0])
    ax4.set_title('Linear Gaussian SSM States (Trial 1)')

    plt.show()


if __name__ == "__main__":
    main()
