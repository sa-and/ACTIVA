"""
Script to visualize encoded/decoded distributions produced by a trained model.

This script loads a dataset and a saved model checkpoint (dill format) and
creates scatter plots of original, encoded, and reconstructed data. It is
intended for offline analysis and produces plots under the `plots/` folder.
"""

from utils import *

import jax
import jax.numpy as jnp

from dataclasses import dataclass
import tyro
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


@dataclass
class Args:
    seed: int = 42
    """Seed for the random number generator used for deterministic sampling."""
    batch_size: int = 1
    """Minibatch size for forming inputs to the model (default 1)."""
    data_mode: str = 'interventional'
    """Whether observational data or interventional data is used."""
    data_path: str = 'data/AAAI/2var_linadd_beta_test.npz'
    """Path to the `.npz` data file to visualize (train/test)."""
    model_path: str = 'models/2var_linadd_beta_fearless_planet'
    """Path to the saved model checkpoint (dill file) used for decoding/encoding visualizations."""


def main():
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    # Load data
    data = load_scm_data(scm_file=args.data_path, mode=args.data_mode)
    samples_to_consider = min(len(data), 500)
    rng, choice_key = jax.random.split(rng)
    data_sample_indicies = jax.random.choice(choice_key, samples_to_consider, shape=(5,), replace=False)
    n_samples = data.arrays[0].shape[1]
    n_targets = data.arrays[-1].shape[-2]
    n_interventions = len(data[1])-3

    # Load model
    model, state = load_model_state(args.model_path)

    if args.data_mode == 'observational':
        raise NotImplementedError('observational data mode not implemented for evaluation')

    elif args.data_mode == 'interventional':
        intervention_indicies = jnp.repeat(jnp.arange(0, model.n_interventions, 1)[jnp.newaxis, :],
                                           samples_to_consider,
                                           axis=0).T
        # Plot original data distribution and save plot
        save_distribution_plot(data.arrays[0], "data/Original Data Distribution", "plots/original_distribution.pdf",
                               tracking=False, has_batch_dim=True)
        save_distribution_plot_colored(data.arrays[0][data_sample_indicies], "data/Original Data Distribution Samples",
                               "plots/original_distribution_samples.pdf",
                               colored=True, tracking=False, has_batch_dim=True)

        # Save encoded data distribution plot
        rng, dropoutkey = jax.random.split(rng)
        encoded_mean = jnp.concat([model.apply({'params': state.params},
                                               x=data.arrays[i+1][:samples_to_consider, j],
                                               dm=data.arrays[0][:samples_to_consider],
                                               intervention_index=intervention_indicies[i],
                                                intervention_targets=data.arrays[-1][:samples_to_consider, i, :, j],
                                               rngs={'dropout': dropoutkey},
                                          method=model.encode, is_training=False)[0] for j in range(n_targets) for i in range(n_interventions)], axis=0)
        save_distribution_plot(encoded_mean, "encodings/media/Encoded Data Distribution", "plots/encoded_distribution.pdf",
 tracking=False)
        save_distribution_plot_colored(encoded_mean[data_sample_indicies], "encodings/media/Encoded Data Distribution Samples",
                               "plots/encoded_distribution_samples.pdf",
                               colored=True, tracking=False)

        # Save reconstructed data distribution plot from latents samples from prior
        rng, sample_rng = jax.random.split(rng)
        dec_distr = [model.apply({'params': state.params}, dm=data.arrays[0][:samples_to_consider],
                                               intervention_index=intervention_indicies[i],
                                                intervention_targets=data.arrays[-1][:samples_to_consider, i, :, j],
                                               rngs={'dropout': dropoutkey},
                                           is_training=False, method=model.inference, rng=sample_rng)
                               for j in range(n_targets) for i in range(len(intervention_indicies))]

        
        rng, samplekey = jax.random.split(rng)
        reconstructed_data = jnp.stack([dec_distr[i].sample(seed=samplekey, sample_shape=(n_samples,)).swapaxes(0, 1) for i in range(len(intervention_indicies)*n_targets)], axis=1)
        

        

    # plot effects of interventions for just a few examples, each color is the original dataset per intervention
    reconstructed_data_selection = reconstructed_data[data_sample_indicies]
    # add original observational data
    reconstructed_data_selection = jnp.concat([reconstructed_data_selection,
           data.arrays[0][data_sample_indicies][:, jnp.newaxis]], axis=1)
    [save_distribution_plot_colored(reconstructed_data_selection[i],
               "reconstruction/Reconstructed Data Distribution Encodings Samples "+str(i),
               "plots/model_reconstructed_samples_"+str(i)+".pdf",
               colored=True, tracking=False, has_batch_dim=True) for i in range(len(reconstructed_data_selection))]


if __name__ == '__main__':
    main()
