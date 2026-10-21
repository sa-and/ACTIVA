import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn
from typing import Tuple

"""
Core model implementations for the project.

This module contains the encoder/decoder and small utility networks used by
the ConditionalVAE training and evaluation scripts. Implementations are
written with Flax (linen) and produce distrax-compatible distribution
parameters (means, covariances, mixture weights) that the rest of the code
base expects.

Key classes:
 - SmallMLP, TransformerBlock, EquivariantNPTEncoder
 - GMMNetwork, TransformerDecoder
 - ConditionalVAE (wires encoder/prior/decoder together)
"""
import functools


def layer_norm(*, reduction_axes, name=None):
    """
    Convenience helper that returns a Flax LayerNorm module configured to
    reduce over the provided axes. This keeps the creation site compact and
    ensures consistent default options (use_scale/use_bias).

    Args:
        reduction_axes: Axis or tuple of axes that LayerNorm will normalize over.
        name: Optional name for the LayerNorm module instance.

    Returns:
        An instance of `flax.linen.LayerNorm`.
    """
    return nn.LayerNorm(reduction_axes=reduction_axes, use_scale=True, use_bias=True, name=name)


def set_diagonal(arr, val):
    """
    Return a new array with the last-two-dimensional diagonal set to `val`.

    This supports broadcasting: if `arr` has shape (..., N, N) the diagonal
    elements arr[..., i, i] are replaced by `val`.

    Args:
        arr: JAX array with shape (..., N, N).
        val: scalar value to write to the diagonal.

    Returns:
        A new JAX array with the diagonal updated.
    """
    n_vars = arr.shape[-1]
    return arr.at[..., jnp.arange(n_vars), jnp.arange(n_vars)].set(val)

class SmallMLP(nn.Module):
    """
    Compact single-hidden-layer MLP used as a small feed-forward block.

    This module applies a Dense -> GeLU -> Dense sequence. It's used across
    the codebase for lightweight projection heads where a full Transformer
    feed-forward would be overkill.

    Attributes:
        out_features: Number of output features for the final Dense layer.
        hidden_dim: Hidden layer width.
        comp_dtype: Computation dtype used for the Dense layers.
    
    Returns:
        The transformed tensor with shape matching the input batch and the
        specified `out_features` in the final channel dimension.
    """
    out_features: int
    hidden_dim: int = 128
    comp_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x= nn.gelu(nn.Dense(self.hidden_dim, dtype=self.comp_dtype, param_dtype=jnp.float32)(x))
        out = nn.Dense(self.out_features, dtype=self.comp_dtype, param_dtype=jnp.float32)(x)
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with pre-layernorm, multi-head self-attention
    and a small feed-forward residual branch.

    Implementation details:
      - Applies LayerNorm before attention (pre-norm)
      - Uses `SmallMLP` as the position-wise feed-forward network
      - Dropout is applied on the attention and FFN outputs when `is_training` is True

    Attributes:
        embedding_dim: q/k/v dimensionality used inside the attention.
        hidden_dim: width of the internal feed-forward MLP.
        attention_heads: number of attention heads.
        dropout: dropout probability.
        comp_dtype: dtype used for computations (defaults to bfloat16 in this code).
    """
    embedding_dim: int
    hidden_dim: int
    attention_heads: int
    dropout: float
    comp_dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x, is_training: bool = True):
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor.
            is_training: Whether the model is in training mode (affects dropout).
            
        Returns:
            Transformed output tensor.
        """
        # Layer normalization
        x_norm = nn.LayerNorm(reduction_axes=-1)(x)

        # Multi-head self-attention
        attn_output = nn.MultiHeadDotProductAttention(
            out_features=x.shape[-1],
            num_heads=self.attention_heads,
            qkv_features=self.embedding_dim,
            dtype=self.comp_dtype,
            param_dtype=jnp.float32
        )(x_norm)

        # Residual connection with dropout
        x = x + nn.Dropout(rate=self.dropout, deterministic=not is_training)(attn_output)

        # Feed-forward network with layer norm
        x_ffn = nn.LayerNorm(reduction_axes=-1)(x)
        x_ffn = SmallMLP(out_features=x.shape[-1], hidden_dim=self.hidden_dim, comp_dtype=self.comp_dtype)(x_ffn)

        # Second residual connection with dropout
        x = x + nn.Dropout(rate=self.dropout, deterministic=not is_training)(x_ffn)

        return x


class EquivariantNPTEncoder(nn.Module):
    """
    Equivariant encoder that follows the Neural Process Transformer pattern.

    The encoder embeds input entries and then alternates attention operations
    across the sample axis and across the variable axis (implemented by
    swapping axes between Transformer blocks). This produces a representation
    that is permutation-equivariant w.r.t. samples and variables.

    Attributes:
        hidden_dim: width used inside Transformer blocks.
        attention_layers: number of transformer blocks (must be even because
                          blocks alternate over axes).
        embedding_dim: size of the per-element embedding.
        dropout: dropout probability.
        attention_heads: number of attention heads.
        n_interventions: number of intervention types; used to shape
                         intervention encodings when creating interventional tensors.
        comp_dtype: computation dtype (bfloat16 by default).
    """
    hidden_dim: int
    attention_layers: int
    embedding_dim: int
    dropout: float
    attention_heads: int
    n_interventions: int
    comp_dtype: jnp.dtype = jnp.bfloat16

    def setup(self) -> None:
        assert self.attention_layers % 2 == 0, "Need an even number of attention layers"

    @nn.compact
    def __call__(self, x, is_training: bool):
        """
        Args:
            x: Input tensor containing embedded data. Expected shape is
               (batch_size, n_samples, n_vars, feature_dim) where the last
               feature dimension contains the element embedding plus any
               intervention encoding.

        Returns:
            A pooled representation tensor with shape (batch_size, n_vars, embedding_dim)
            that can be further aggregated by the encoder.
        """
        # Embed each element of the input. As extension an element-wise linear layer with shared weights per column
        # could be implemented here as suggested in Kossen et al.
        H = nn.Dense(features=self.embedding_dim, dtype=self.comp_dtype, param_dtype=jnp.float32)(x)

        for _ in range(self.attention_layers):
            # mha
            MHSA_H = TransformerBlock(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                attention_heads=self.attention_heads,
                dropout=self.dropout
            )(H, is_training=is_training)

            # flip N and d axes
            H = jnp.swapaxes(MHSA_H, -2, -3)

        z = nn.LayerNorm(reduction_axes=-1)(H)

        z = jnp.mean(z, axis=-3)

        return z

class MVNDiagNetwork(nn.Module):
    """
    Network that outputs parameters for a multivariate normal distribution with diagonal covariance.

    NOTE: This class is not used by the current ConditionalVAE implementation.
    The encoder now uses a plain `nn.Sequential` (`q_conditioner`) with a fixed
    log-variance of 0.1. This class is kept for reference.

    Attributes:
        comp_dtype: Data type for computations.
        hidden_dim: Dimension of the hidden layer.
    """
    comp_dtype: jnp.dtype = jnp.bfloat16
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x):
        """Project inputs to a mean and log-variance vector.

        Args:
            x: Input tensor with shape (..., features) or (..., positions, features).

        Returns:
            A tuple (means, logvars) where both tensors have shape
            (..., latent_dim). In the current code latent_dim will usually be 1
            and calling sites reshape/aggregate as needed.
        """
        # Mean vector for the distribution
        means = SmallMLP(out_features=1, hidden_dim=self.hidden_dim,  comp_dtype=self.comp_dtype)(x)
        means = means.reshape(means.shape[:-2] + (-1,))
        logvars = SmallMLP(out_features=1, hidden_dim=self.hidden_dim, comp_dtype=self.comp_dtype)(x)
        logvars = logvars.reshape(logvars.shape[:-2] + (-1,))

        return means, logvars

class GMMNetwork(nn.Module):
    """
    Network that outputs parameters for a Gaussian mixture model over the
    observed data space.

    Returns three tensors: `weights`, `means`, and `covs` with shapes:
      - weights: (batch_size, num_components)
      - means:   (batch_size, num_components, data_dim)
      - covs:    (batch_size, num_components, data_dim, data_dim)

    The implementation builds per-position component parameters and pools
    across positions to produce sample-level mixture logits (weights).

    Attributes:
        comp_dtype: Data type for computations.
        num_components: Number of mixture components.
        embedding_dim: Dimension of the embedding.
    """
    comp_dtype: jnp.dtype = jnp.bfloat16
    num_components: int = 1
    embedding_dim: int = 32
    compute_covs: bool = True

    @nn.compact
    def __call__(self, x):
        # Compute per-component means and covariance factors per position
        means = SmallMLP(out_features=self.num_components,
                         hidden_dim=self.embedding_dim,
                         comp_dtype=self.comp_dtype)(x)  # Shape: (batch_size, data_dim, num_components)
        means = jnp.transpose(means, (0, 2, 1))  # Shape: (batch_size, num_components, data_dim)

        # Mixture weights (per sample, not per position)
        # Use a pooling operation that is permutation invariant
        z_pooled = jnp.sum(x, axis=1)  # Shape: (batch_size, embedding_dim)
        weights = SmallMLP(out_features=self.num_components,
                           hidden_dim=self.embedding_dim,
                           comp_dtype=self.comp_dtype)(z_pooled)  # Shape: (batch_size, num_components)
        weights = nn.softmax(weights.astype(jnp.float32)).astype(jnp.bfloat16)

        if self.compute_covs:
            u = SmallMLP(out_features=self.num_components * self.embedding_dim,
                         hidden_dim=self.embedding_dim,
                         comp_dtype=jnp.float32)(x)

            u = u.reshape(*u.shape[:-1], self.num_components, self.embedding_dim)
            u = jnp.transpose(u, (0, 2, 1, 3))
            # compute log variances and non-log covariances
            covs = jnp.einsum("...id,...jd->...ij", u, u)
            vars = SmallMLP(out_features=self.num_components, hidden_dim=self.embedding_dim, comp_dtype=self.comp_dtype)(x)
            vars = jnp.transpose(vars, (0, 2, 1))
            vars = nn.softplus(vars.astype(jnp.float32)).astype(jnp.bfloat16) + 1e-3
            eye = jnp.eye(vars.shape[-1])[None, None, :, :]
            vars = vars[..., None] * eye
            covs = covs + vars
        else:
            covs = None

        return weights, means, covs

class TransformerDecoder(nn.Module):
    """
    Decoder that maps latent per-variable representations into a GMM over the
    observed data space using Transformer blocks followed by a GMM head.

    The decoder accepts a per-variable latent tensor and returns mixture
    weights, component means and full covariance matrices for each component.

    Attributes mirror those of the encoder (embedding_dim, hidden_dim, etc.).
    """
    data_dim: int
    num_components: int
    hidden_dim: int
    attention_layers: int
    dropout: float
    embedding_dim: int
    attention_heads: int
    comp_dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, decoder_input, is_training: bool = True):
        """
        Transformer-based decoder that maintains permutation equivariance.

        Args:
            decoder_input: Latent input shaped (batch_size, n_vars, latent_dim)
                           where latent_dim typically equals 1 + n_interventions.
            is_training: boolean flag passed to Transformer layers (controls
                         dropout behavior).

        Returns:
            weights: Mixture weights tensor of shape (batch_size, num_components).
            means:   Component means tensor of shape (batch_size, num_components, data_dim).
            covs:    Component covariance matrices of shape
                     (batch_size, num_components, data_dim, data_dim).
        """
        z = nn.Dense(self.embedding_dim, dtype=self.comp_dtype, param_dtype=jnp.float32)(decoder_input)

        # Transformer layers without positional embeddings
        for _ in range(self.attention_layers):
            z = TransformerBlock(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                attention_heads=self.attention_heads,
                dropout=self.dropout
            )(z, is_training=is_training)

        z = nn.LayerNorm(reduction_axes=-1)(z)


        weights, means, covs = GMMNetwork(
                                        num_components=self.num_components,
                                          embedding_dim=self.embedding_dim,
                                          comp_dtype=self.comp_dtype
                                        )(z)
        return weights, means, covs


class ConditionalVAE(nn.Module):
    """
    Conditional VAE specialized for modeling interventional distributions.

    Important notes about the current implementation:
      - The encoder produces a mean vector and a fixed log-variance (constant 0.1).
      - The prior is implemented as a small GMM over the encoded observational data.
      - The decoder outputs a Gaussian mixture (weights, means, covariances) per batch.

    This class wires encoder/prior/decoder modules together and exposes a
    convenient `inference` method that samples from the prior and decodes to
    produce an interventional distribution object (a distrax MixtureSameFamily).
    """
    n_interventions: int
    data_dim: int
    n_samples: int  # number of samples per input
    hidden_dim: int
    embedding_dim: int
    dropout: float
    attention_heads: int
    encoder_attention_layers: int
    decoder_attention_layers: int
    decoder_mixture_components: int
    prior_mixture_components: int
    comp_dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.int_data_network = EquivariantNPTEncoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                                      dropout=self.dropout, attention_heads=self.attention_heads,
                                                      attention_layers=self.encoder_attention_layers,
                                                      n_interventions=self.n_interventions,
                                                      comp_dtype=self.comp_dtype)
        self.obs_data_network = EquivariantNPTEncoder(embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
                                                      dropout=self.dropout, attention_heads=self.attention_heads,
                                                      attention_layers=self.encoder_attention_layers,
                                                      n_interventions=self.n_interventions,
                                                      comp_dtype=self.comp_dtype)
        self.q_conditioner = nn.Sequential((nn.Dense(self.hidden_dim),
                                            nn.gelu,
                                            nn.Dense(self.hidden_dim),
                                            nn.gelu,
                                            nn.Dense(1)))
        self.p_gmm_head = GMMNetwork(num_components=self.prior_mixture_components,
                                     embedding_dim=self.embedding_dim,
                                     comp_dtype=self.comp_dtype,
                                     compute_covs=False)
        self.decoder_network = TransformerDecoder(data_dim=self.data_dim, embedding_dim=self.embedding_dim,
                                                  hidden_dim=self.hidden_dim, dropout=self.dropout,
                                                  attention_heads=self.attention_heads,
                                                  attention_layers=self.decoder_attention_layers,
                                                  num_components=self.decoder_mixture_components,
                                                  comp_dtype=self.comp_dtype)

    def __call__(self, x: jnp.ndarray, dm: jnp.ndarray, intervention_index: jnp.ndarray,
                 intervention_targets: jnp.ndarray, is_training: bool, rng) \
            -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Forward pass that encodes inputs, samples latents (via reparameterize)
        and decodes to output distribution parameters.

        Args:
            x: Interventional batch array with shape (batch, samples, n_vars).
            dm: Observational dataset (used to form the conditional prior).
            intervention_index: Integer index or array indicating intervention type per batch.
            intervention_targets: Binary mask indicating which variables are intervened on.
            is_training: toggles dropout and training-only behaviour.
            rng: PRNG key for sampling operations.

        Returns:
            (weights, means, covs, z_mean_hat, z_logvars)
            where weights/means/covs parameterize the decoded mixture distribution,
            and z_mean_hat/z_logvars are the encoder outputs used for sampling.
        """
        # Step 1: Encode the input data to get distribution parameters
        z_mean_hat, z_logvars = self.encode(x, dm, intervention_index, intervention_targets, is_training)

        # Step 2: Apply the reparameterization trick to sample from the latent space
        z_reparam = self.reparameterize(z_mean_hat, z_logvars, rng)

        # Step 3: Decode the latent sample to get parameters of the output distribution
        weights, means, covs = self.decode(z_reparam, is_training)

        return weights, means, covs, z_mean_hat, z_logvars

    #@functools.partial(jax.jit, static_argnames=['samples_from_prior'])
    def inference(self, dm: jnp.ndarray, intervention_index: jnp.ndarray,
                 intervention_targets: jnp.ndarray, rng, samples_from_prior:int=5, is_training: bool = False) \
            -> Tuple[distrax.Distribution, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate an approximate interventional distribution by sampling the model's
        conditional prior and decoding the samples.

        The method draws `samples_from_prior` samples from the conditional prior for
        each batch element, decodes each latent into a GMM in data-space, and
        combines the decoded mixtures into a single higher-order mixture which
        approximates the marginal interventional distribution.

        Behavior and shapes:
          - `dm` is expected to have shape (batch_size, n_samples, n_vars) when
            evaluating a batch of conditional priors.
          - The returned object is a `distrax.MixtureSameFamily` instance whose
            component distribution is itself a `distrax.MixtureSameFamily`.
            This two-level mixture encodes the sampling-from-prior-then-decode
            construction used here.

        Args:
            dm: Observational dataset used to construct the conditional prior.
            intervention_index: intervention index (or array) to condition the prior.
            intervention_targets: mask of target variables for the intervention.
            rng: PRNGKey for sampling.
            samples_from_prior: number of prior samples to draw per batch element.
            is_training: passed to encoder/decoder (kept False for inference by default).

        Returns:
            A `distrax` distribution object representing the decoded interventional
            mixture. See note above about the nested MixtureSameFamily structure.
        """
        # Step 1: Encode observational data to get prior distribution parameters
        pri_weights, pri_means, pri_covs = self.encode_prior(dm=dm,
                                                            intervention_index=intervention_index,
                                                            intervention_targets=intervention_targets,
                                                            is_training=is_training)

        # Step 2: Compute prior distribution and sample from it
        prior_test_distribution = self.get_conditional_prior(weights=pri_weights,
                                                            means=pri_means,
                                                            covs=pri_covs)
        rng, samplekey = jax.random.split(rng)
        prior_samples = prior_test_distribution.sample(seed=samplekey, sample_shape=(samples_from_prior,))

        # Step 3: Decode samples to get parameters of the output distribution
        dec_weights, dec_means, dec_covs = jax.vmap(lambda z: self.decode(z, is_training=False))(prior_samples)
        
        # Step 4: Create a uniform mixture of the decoded distributions - extract dims
        b = prior_samples.shape[1]  # batch size
        s = prior_samples.shape[0]  # number of samples per batch
        # Reshape the decoded parameters and assemble the two-level mixture.
        # The active implementation below creates a MixtureSameFamily where the outer
        # mixture selects a sample and the inner mixture represents decoder components.
        decod_distribution = distrax.MixtureSameFamily(distrax.Categorical(probs=jnp.ones((b, s )) / (s )),
                                                distrax.MixtureSameFamily(distrax.Categorical(probs=dec_weights.swapaxes(0,1)),
                                                                            distrax.MultivariateNormalFullCovariance(loc=dec_means.swapaxes(0,1), covariance_matrix=dec_covs.swapaxes(0,1))))
        
        return decod_distribution

    def init_networks(self, x, dm, intervention_index, intervention_targets, is_training, rng):
        """
        Special method to ensure all networks in the model are properly initialized.
        
        This method calls both the forward pass and the prior encoding to make sure
        all parameters in all networks are initialized, including those that might
        not be used in every forward pass.
        
        Args:
            x: Interventional data sample.
            dm: Observational dataset.
            intervention_index: Index indicating which intervention was used.
            intervention_targets: Binary vector indicating which variables are targets of the intervention.
            is_training: Whether the model is in training mode.
            rng: Random number generator key for sampling.
            
        Returns:
            The outputs from the forward pass.
        """
        # Step 1: Call the regular forward pass to initialize encoder and decoder
        outputs = self(x, dm, intervention_index, intervention_targets, is_training, rng)

        # Step 2: Call the prior encoding to make sure it's initialized
        _ = self.encode_prior(dm, intervention_index, intervention_targets, is_training)

        return outputs

    def forward_with_prior(self, x, dm, intervention_index, intervention_targets, is_training: bool, rng):
        """
        Combined forward pass that calls obs_data_network exactly once and uses
        the result for both the posterior encoder and the prior GMM head.

        This is mathematically identical to calling encode() + encode_prior() separately
        (both use make_interventional_tensor(dm,...) → obs_data_network as input), but
        eliminates the redundant second call to obs_data_network, halving encoder compute.

        Gradient equivalence: JAX autodiff accumulates ∂L/∂z_obs from both the posterior
        and prior branches through the shared activation, producing the same gradient as
        two separate calls with identical inputs.

        Returns:
            (dec_weights, dec_means, dec_covs, z_mean_hat, z_logvars,
             pri_weights, pri_means, pri_covs)
        """
        # Build the intervention mask once and reuse for both dm and x tensors
        intervention_mask = self._make_intervention_mask(intervention_index, intervention_targets, dm.shape[1])

        # Single obs_data_network call — shared by both posterior and prior branches
        encoder_input = self._apply_intervention_mask(dm, intervention_mask)
        z_obs = self.obs_data_network(encoder_input, is_training)

        # Prior branch: identical to encode_prior()
        pri_weights, pri_means, _ = self.p_gmm_head(z_obs)
        data_dim = pri_means.shape[-1]
        pri_covs = jnp.broadcast_to(jnp.eye(data_dim) * 0.1, pri_means.shape + (data_dim,))

        # Posterior branch: identical to encode()
        int_data_input = self._apply_intervention_mask(x, intervention_mask)
        z_int = self.int_data_network(int_data_input, is_training)
        conditioned_means = jnp.concatenate([z_obs, z_int], axis=-1)
        z_mean_hat = self.q_conditioner(conditioned_means).squeeze()
        z_logvars = jnp.ones_like(z_mean_hat) * 0.1

        # Reparameterize and decode: identical to __call__()
        z_reparam = self.reparameterize(z_mean_hat, z_logvars, rng)
        dec_weights, dec_means, dec_covs = self.decode(z_reparam, is_training)

        return dec_weights, dec_means, dec_covs, z_mean_hat, z_logvars, pri_weights, pri_means, pri_covs

    def encode(self, x, dm, intervention_index, intervention_targets, is_training: bool):
        """
        Encodes the input data into the latent space.
        
        This method:
        1. Processes observational data through the encoder network
        2. Processes interventional data through the encoder network
        3. Concatenates and conditions the encodings to get the final latent representation
        
        Args:
            x: Interventional data sample.
            dm: Observational dataset.
            intervention_index: Index indicating which intervention was used.
            intervention_targets: Binary vector indicating which variables are targets of the intervention.
            is_training: Whether the model is in training mode.
            
        Returns:
            A tuple containing:
            - means: Mean of the latent distribution.
            - logvars: Log variances of the latent distribution (fixed value in this implementation).
        """
        # Step 1: Process observational data through the encoder network
        encoder_input = self.make_interventional_tensor(dm, intervention_index, intervention_targets)
        z_obs = self.obs_data_network(encoder_input, is_training)

        # Step 2: Process interventional data through the encoder network
        int_data_input = self.make_interventional_tensor(x, intervention_index, intervention_targets)
        z_int = self.int_data_network(int_data_input, is_training)

        # Step 3: Concatenate and condition the encodings
        conditioned_means = jnp.concatenate([z_obs, z_int], axis=-1)
        means = self.q_conditioner(conditioned_means).squeeze()
        
        # Return means and fixed log variances (will be used in get_encoded_distribution)
        return means, jnp.ones_like(means)*0.1

    def decode(self, z: jnp.ndarray, is_training: bool) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Decodes a latent vector into parameters for the output distribution.
        
        This method:
        1. Expands the latent vector to match the expected input shape for the decoder
        2. Processes the expanded vector through the decoder network
        
        Args:
            z: Latent vector with shape (batch_size, n_vars).
            is_training: Whether the model is in training mode.
            
        Returns:
            A tuple containing:
            - weights: Mixture weights for the output distribution.
            - means: Means for the output distribution.
            - covs: Covariance matrices for the output distribution.
        """
        # Step 1: Expand z to match the data_dim dimension expected by the decoder
        z_expanded = z[..., jnp.newaxis]  # Shape: (batch_size, n_var, 1)

        # Step 2: Process through the decoder network
        weights, means, covs = self.decoder_network(z_expanded, is_training=is_training)
        return weights, means, covs

    def encode_prior(self, dm, intervention_index, intervention_targets, is_training: bool):
        """
        Encodes observational data to generate parameters for the prior distribution.
        
        This method:
        1. Processes observational data through the encoder network
        2. Generates mixture weights, means, and covariance matrices for the prior
        3. Adjusts covariance matrices to match the encoder's fixed variance
        
        Args:
            dm: Observational dataset.
            intervention_index: Index indicating which intervention to model.
            intervention_targets: Binary vector indicating which variables are targets of the intervention.
            is_training: Whether the model is in training mode.
            
        Returns:
            A tuple containing:
            - weights: Mixture weights for the prior distribution.
            - means: Means for the prior distribution.
            - covs: Covariance matrices for the prior distribution.
        """
        # Step 1: Process observational data through the encoder network
        prior_input = self.make_interventional_tensor(dm, intervention_index, intervention_targets)
        z = self.obs_data_network(prior_input, is_training)
        
        # Step 2: Generate mixture parameters from the encoded representation
        weights, means, _ = self.p_gmm_head(z)

        # Step 3: Build fixed 0.1*I covariance from means shape (covs not computed by p_gmm_head)
        data_dim = means.shape[-1]
        covs = jnp.broadcast_to(jnp.eye(data_dim) * 0.1, means.shape + (data_dim,))

        return weights, means, covs

    def get_encoded_distribution(self, mean, logvar) -> distrax.Distribution:
        """
        Creates a distribution object for the encoded latent representation.
        
        This method creates a multivariate normal distribution with diagonal covariance
        based on the mean and a fixed standard deviation of 0.1.
        
        Args:
            mean: Mean vector of the distribution.
            logvar: Log variance vector (not used in this implementation).
            
        Returns:
            A MultivariateNormalDiag distribution object.
        """
        # Create a multivariate normal distribution with diagonal covariance
        # Fixed variance of 0.1 (matching the prior's 0.1·I covariance)
        q_z_given_x_u = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.ones_like(logvar)*jnp.sqrt(0.1)
        )
        return q_z_given_x_u

    def get_decoded_distr(self, weights: jnp.ndarray, means: jnp.ndarray, covs: jnp.ndarray) -> distrax.Distribution:
        """
        Creates a mixture distribution from the decoded parameters.
        
        This method creates a mixture of multivariate normal distributions with full covariance
        matrices based on the provided weights, means, and covariance matrices.
        
        Args:
            weights: Mixture weights with shape (batch_size, num_components).
            means: Component means with shape (batch_size, num_components, data_dim).
            covs: Component covariance matrices with shape (batch_size, num_components, data_dim, data_dim).
            
        Returns:
            A MixtureSameFamily distribution object.
        """
        # Create a mixture of multivariate normal distributions
        # Note: This reduces to a single multivariate normal distribution when components = 1
        distr = distrax.MixtureSameFamily(
            mixture_distribution=distrax.Categorical(probs=weights),
            components_distribution=distrax.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covs)
        )
        return distr

    def get_conditional_prior(self, weights: jnp.ndarray, means: jnp.ndarray, covs: jnp.ndarray):
        """
        Creates the conditional prior distribution.
        
        This method creates a mixture distribution for the prior using the same
        distribution type as the decoder output. It uses the provided weights,
        means, and covariance matrices from the encode_prior method.
        
        Args:
            weights: Mixture weights with shape (batch_size, num_components).
            means: Component means with shape (batch_size, num_components, data_dim).
            covs: Component covariance matrices with shape (batch_size, num_components, data_dim, data_dim).
            
        Returns:
            A MixtureSameFamily distribution object representing the conditional prior.
        """
        # Use the same distribution type as the decoder output
        return self.get_decoded_distr(weights, means, covs)

    def reparameterize(self, z_mean: jnp.ndarray, z_logvar: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Apply the reparameterization trick to sample from the latent distribution.

        For numerical stability and simplicity this implementation uses a fixed
        standard deviation of 0.1 (instead of using exp(0.5 * z_logvar)). The
        returned sample has the same shape as `z_mean`.

        Args:
            z_mean: Mean of the latent distribution, shape (batch_size, n_vars) or similar.
            z_logvar: Log variances of the latent distribution (not used here, but kept
                      for API compatibility with other MVN heads).
            rng: PRNG key used to generate Gaussian noise.

        Returns:
            A JAX array containing a sample from N(z_mean, 0.1^2) with the same
            shape and dtype as `z_mean`.
        """
        # Generate noise and scale by fixed std = sqrt(0.1) to match the prior's 0.1·I covariance
        epsilon = jax.random.normal(rng, shape=z_mean.shape)
        return z_mean + jnp.astype(jnp.sqrt(0.1) * epsilon, self.comp_dtype)

    def _make_intervention_mask(self, intervention_index: jnp.ndarray,
                                intervention_targets: jnp.ndarray,
                                n_samples: int) -> jnp.ndarray:
        """
        Build the masked intervention encoding that is independent of the data tensor.

        Returns:
            Tensor of shape (batch_size, n_samples, n_vars, n_interventions) containing
            the one-hot intervention encoding masked by intervention_targets, broadcast
            across samples. This can be reused across different data tensors (dm, x).
        """
        # One-hot encoding: (batch_size, n_interventions)
        intervention_rep = nn.one_hot(intervention_index,
                                      num_classes=self.n_interventions, dtype=self.comp_dtype)

        # Broadcast to (batch_size, n_samples, n_vars, n_interventions) without copying
        batch_size = intervention_rep.shape[0]
        n_vars = intervention_targets.shape[-1]
        # (batch, 1, 1, n_interventions)
        intervention_expanded = jnp.broadcast_to(
            intervention_rep[:, jnp.newaxis, jnp.newaxis, :],
            (batch_size, n_samples, n_vars, self.n_interventions)
        )

        # Targets mask: (batch, 1, n_vars, 1) → broadcasts over samples and interventions
        targets_expanded = intervention_targets[:, jnp.newaxis, :, jnp.newaxis]

        return intervention_expanded * targets_expanded

    def _apply_intervention_mask(self, x: jnp.ndarray,
                                  intervention_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Concatenate data with a precomputed intervention mask.

        Args:
            x: Data tensor of shape (batch_size, n_samples, n_vars).
            intervention_mask: Precomputed mask from _make_intervention_mask,
                shape (batch_size, n_samples, n_vars, n_interventions).

        Returns:
            Tensor of shape (batch_size, n_samples, n_vars, 1 + n_interventions).
        """
        return jnp.concatenate([x[..., jnp.newaxis], intervention_mask], axis=-1)

    def make_interventional_tensor(self, x: jnp.ndarray, intervention_index: jnp.ndarray,
                                intervention_targets: jnp.ndarray) -> jnp.ndarray:
        """
        Create an input tensor that augments raw data with an intervention encoding.

        The encoder expects per-element features with the final channel containing
        the original observed value plus a one-hot intervention indicator. This
        method constructs that tensor and applies `intervention_targets` as a mask
        so the one-hot is only present for intervened variables.

        Args:
            x: Input data tensor with shape (batch_size, samples, n_vars).
            intervention_index: Integer index (or per-example array) indicating which
                                intervention to model; shape (batch_size,) or ().
            intervention_targets: Binary mask with shape (batch_size, n_vars) where
                                a 1 indicates the corresponding variable is targeted.

        Returns:
            A float tensor with shape (batch_size, samples, n_vars, 1 + n_interventions)
            where the last channel contains [value, one_hot_intervention...]. The
            one-hot channels are zeroed for non-target variables according to
            `intervention_targets`.
        """
        intervention_mask = self._make_intervention_mask(intervention_index, intervention_targets, x.shape[1])
        return self._apply_intervention_mask(x, intervention_mask)

