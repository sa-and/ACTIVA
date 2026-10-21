"""
Plot baseline distributions for comparison with model-generated samples.

This helper script reads baseline samples and the original data and renders
side-by-side visualizations so practitioners can compare reconstruction
quality and distributional properties.
"""

from utils import *

import jax
import jax.numpy as jnp
import numpy

from dataclasses import dataclass
import tyro
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


@dataclass
class Args:
    seed: int = 42
    """Seed for the random number generator used for deterministic baseline plots."""
    data_path: str = ''
    """Path to the `.npz` dataset used to compute and plot baseline distributions."""


def main():
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    # Load data
    data = load_scm_data(scm_file=args.data_path, mode='interventional')
    samples_to_consider = min(len(data), 500)
    scms_to_plot = 10
    rng, choice_key = jax.random.split(rng)
    data_sample_indicies = [i for i in range(scms_to_plot)]
    n_samples = data.arrays[0].shape[1]
    n_targets = data.arrays[-1].shape[-2]
    n_interventions = len(data[1])-3
    interv_value = 5.0
    input_data = data.arrays[0]

    # Plot original data distribution and save plot
    save_distribution_plot(data.arrays[0], "data/Original Data Distribution", "plots/base_original_distribution.pdf",
                           tracking=False, has_batch_dim=True)
    save_distribution_plot_colored(data.arrays[0][data_sample_indicies], "data/Original Data Distribution Samples",
                           "plots/base_original_samples.pdf",
                           colored=True, tracking=False, has_batch_dim=True)

    
    not_converged = 0
    all_samples=[]
    for test_index in data_sample_indicies:  # do with 1000 iter
        try:
            means, covs = fit_multivariate_gaussian(input_data[test_index])
            obs_samples = sample_gaussian(means, covs, input_data.shape[1])
            print('obs est', str(means))
        except numpy.linalg.LinAlgError as e:
            print(test_index, e)
            not_converged += 1
            continue

        estimated_samples = []
        for i in range(len(data.arrays[1:-2])):
            for j in range(data.arrays[i + 1].shape[1]):
                cond_means, cond_covs = conditional_gaussian_params(means, covs, j, interv_value)
                print(str(i), str(j), str(cond_means))
                try:
                    samples = sample_gaussian(cond_means, cond_covs, input_data.shape[1])
                except numpy.linalg.LinAlgError as e:
                    print(test_index, i, j, e)
                    not_converged += 1
                    continue

                samples = jnp.concat([samples[:, :j], jnp.ones((samples.shape[0], 1)) * interv_value, samples[:, j:]],
                                     axis=-1)
                estimated_samples.append(samples)
        estimated_samples.append(jnp.array(obs_samples))
        all_samples.append(estimated_samples)

    [save_distribution_plot_colored(jnp.array(all_samples[i]),
                           "reconstruction/Reconstructed Data Distribution Encodings Samples "+str(i),
                           "plots/base_reconstructed_samples_"+str(i)+".pdf",
                           colored=True, tracking=False, has_batch_dim=True) for i in range(len(all_samples))]

    all_obs_samples = jnp.stack([jnp.concat([data.arrays[i + 1][t], input_data[t][jnp.newaxis,...]], axis=0) for t in data_sample_indicies])
    # save original samples
    [save_distribution_plot_colored(all_obs_samples[i],
                           "data/Original Data Distribution Samples",
                           "plots/original_samples_" + str(i) + ".pdf",
                           colored=True, tracking=False, has_batch_dim=True) for i in range(len(all_obs_samples))]


if __name__ == '__main__':
    main()