from jax import vmap
import jax.random as jr

def generate_data_from_model(model, params, key, trials, num_timesteps):
    keys = jr.split(key, trials)
    sample_many_trials = vmap(model.sample, (None, 0, None), (0, 0))
    true_states, emissions = sample_many_trials(
        params, keys, num_timesteps
    )
    return true_states, emissions