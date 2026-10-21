"""
MACE-TNP model extension: Joint multivariate GMM predictor over all n_vars.

Extends the MACE-TNP architecture from CausalInferenceNeuralProcess (CITNP)
to output a joint K-component full-covariance Gaussian Mixture Model over
all n_vars dimensions simultaneously, rather than per-variable marginals.

Architecture:
    - Encoder: MACE (from CITNP's BaseCausalInference) — Transformer-based,
      permutation-equivariant, handles context + masked target.
    - Predictor: MultivariateMoGPredictor — mean-pools node embeddings, then
      decodes to GMM with full Cholesky covariance per component.

Dependencies:
    pip install torch causina pytorch_lightning einops
    cd ~/CausalInferenceNeuralProcess && pip install -e .
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import (
    Categorical,
    MultivariateNormal,
    MixtureSameFamily,
)

# CITNP imports — available after: pip install -e ~/CausalInferenceNeuralProcess
from functools import partial

from CITNP.models.causalinferencemodel import (
    BaseCausalInference,
    CausalInfModelConfig,
)
from CITNP.models.causaltransformercomponents import CausalInfTransformerEncoder
from CITNP.models.nncomponents import build_mlp
from CITNP.networks.attention_layers import (
    MultiHeadCrossAttentionLayer,
    MultiHeadSelfAttentionLayer,
)
from CITNP.utils.utils import _get_clones


@dataclass
class JointModelOutput:
    """Output of JointMoGCausalInferenceModel.forward()."""
    means: torch.Tensor       # (batch, K, D)
    L: torch.Tensor           # (batch, K, D, D) lower-triangular Cholesky
    weights: torch.Tensor     # (batch, K) after softmax
    target_data: torch.Tensor # (batch, n_tgt, D) passed through for loss


class MultivariateMoGPredictor(nn.Module):
    """
    Maps a set of per-node latent encodings to a joint K-component GMM.

    Takes node embeddings of shape (batch, n_vars, d_model), mean-pools
    over the node axis, then decodes to GMM parameters:
        - means:   (batch, K, D)
        - L:       (batch, K, D, D) lower-triangular (Cholesky of covariance)
        - weights: (batch, K)  (after softmax)

    The diagonal of L uses softplus + 1e-4 for numerical positive-definiteness.
    """

    def __init__(self, d_model: int, n_vars: int, n_components: int, hidden_dim: int = 256):
        super().__init__()
        self.n_components = n_components
        self.n_vars = n_vars
        D = n_vars
        K = n_components
        tril_size = D * (D + 1) // 2

        # Per-node mean head: each node's d_model encoding → K scalar means for that variable
        # This preserves per-variable causal signals instead of mixing them via mean-pooling
        self.per_node_mean_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, K),
        )

        # Shared backbone for covariance/weights (mean-pool is OK for joint structure)
        self.backbone = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Output heads for covariance and mixture weights
        self.chol_head   = nn.Linear(hidden_dim, K * tril_size)
        self.weight_head = nn.Linear(hidden_dim, K)

        # Pre-compute indices for building lower-triangular matrix
        self._tril_row, self._tril_col = torch.tril_indices(D, D)

    def forward(self, node_encodings: torch.Tensor):
        """
        Args:
            node_encodings: (batch, n_vars, d_model)

        Returns:
            means:   (batch, K, D)
            L:       (batch, K, D, D) lower-triangular Cholesky
            weights: (batch, K)
        """
        B = node_encodings.shape[0]
        D = self.n_vars
        K = self.n_components
        tril_size = D * (D + 1) // 2

        # Per-node mean prediction: each node's encoding → K means for that variable
        # Shape: (batch, n_vars, K) → permute → (batch, K, n_vars)
        means = self.per_node_mean_head(node_encodings).permute(0, 2, 1)  # (batch, K, D)

        # Mean-pool over nodes for covariance and weight prediction
        h = node_encodings.mean(dim=1)       # (batch, d_model)
        h = self.backbone(h)                  # (batch, hidden_dim)

        # Decode Cholesky lower-triangular factors
        L_flat = self.chol_head(h).view(B, K, tril_size)  # (batch, K, tril_size)
        L = torch.zeros(B, K, D, D, device=h.device, dtype=h.dtype)

        row = self._tril_row.to(h.device)
        col = self._tril_col.to(h.device)
        L[:, :, row, col] = L_flat

        # Softplus + eps on diagonal for positive-definiteness
        diag_idx = torch.arange(D, device=h.device)
        L[:, :, diag_idx, diag_idx] = torch.nn.functional.softplus(
            L[:, :, diag_idx, diag_idx]
        ) + 0.1

        # Mixture weights
        weights = torch.softmax(self.weight_head(h), dim=-1)  # (batch, K)

        return means, L, weights


class OutcomeFreeCausalInfEncoder(nn.Module):
    """
    Encoder with two roles only: variable and intervention.

    Identical to CITNP's CausalInfEncoder except the outcome role is removed.
    Embedding rule:
        embedding = var_emb * (1 - mask_int) + int_emb * mask_int

    This avoids the conceptual issue of requiring a designated scalar outcome
    index when we want to predict the full joint distribution.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        num_nodes: int,
        device,
        dtype,
        emb_depth: int = 1,
        dropout: float = 0.0,
        sample_attn_mode: str = "MHSA",
        linear_attention: bool = False,
    ):
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        super().__init__()

        mlp = partial(
            build_mlp,
            dim_in=1,
            dim_hid=d_model,
            dim_out=d_model,
            depth=emb_depth,
        )
        self.context_variable     = mlp()
        self.context_intervention = mlp()
        self.target_variable      = mlp()
        self.target_intervention  = mlp()

        module_mhsa = partial(
            MultiHeadSelfAttentionLayer,
            embed_dim=d_model,
            num_heads=nhead,
            head_dim=d_model // nhead,
            feedforward_dim=dim_feedforward,
            p_dropout=dropout,
            norm_first=True,
            activation=nn.GELU(),
            linear=linear_attention,
        )
        if sample_attn_mode == "MHCA":
            module_mhca = partial(
                MultiHeadCrossAttentionLayer,
                embed_dim=d_model,
                num_heads=nhead,
                head_dim=d_model // nhead,
                feedforward_dim=dim_feedforward,
                p_dropout=dropout,
                norm_first=True,
                activation=nn.GELU(),
                linear=linear_attention,
            )
            sample_layers = _get_clones((module_mhsa, module_mhca), num_layers // 2)
        elif sample_attn_mode == "MHSA":
            sample_layers = _get_clones(module_mhsa, num_layers // 2)
        else:
            raise ValueError(
                f"Unknown sample attention mode: {sample_attn_mode}. "
                "Use either 'MHSA' or 'MHCA'."
            )

        node_layers = _get_clones(module_mhsa, num_layers // 2)
        self.encoder = CausalInfTransformerEncoder(
            sample_layers=sample_layers,
            node_layers=node_layers,
            sample_attn_mode=sample_attn_mode,
        )

    def embed(self, data: torch.Tensor, intervention_indices: torch.Tensor, context: bool = True) -> torch.Tensor:
        """
        Args:
            data:                 (batch, n_samples, n_nodes, 1)
            intervention_indices: (batch,) int64
            context:              True for context data, False for target data
        Returns:
            embedding: (batch, n_samples, n_nodes, d_model)
        """
        batch_size = data.size(0)

        if context:
            var_emb = self.context_variable(data)
            int_emb = self.context_intervention(data)
        else:
            var_emb = self.target_variable(data)
            int_emb = self.target_intervention(data)

        batch_idx = torch.arange(batch_size, device=data.device)
        mask_int = torch.zeros_like(var_emb)
        mask_int[batch_idx[:, None], :, intervention_indices[:, None], :] = 1.0

        return var_emb * (1 - mask_int) + int_emb * mask_int

    def create_input(self, context_embed, target_embed, target_train_embed):
        if self.training and target_train_embed is not None:
            return torch.cat([context_embed, target_train_embed, target_embed], dim=1)
        return torch.cat([context_embed, target_embed], dim=1)

    def create_sample_mask(self, context_embed, target_embed, target_train_embed):
        num_context = context_embed.size(1)
        num_target  = target_embed.size(1)
        if target_train_embed is not None:
            mask_size = num_context + num_target + num_target
        else:
            mask_size = num_context + num_target
        sample_mask = torch.zeros(
            mask_size, mask_size,
            device=context_embed.device,
            dtype=context_embed.dtype,
        ).fill_(float("-inf"))
        sample_mask[:, :num_context] = 0.0
        return sample_mask, num_target

    def encode(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_indices: torch.Tensor,
        target_train_data: torch.Tensor | None = None,
        variable_mask=None,
    ):
        """
        Args:
            context_data:         (batch, n_ctx,  n_nodes, 1)
            target_data:          (batch, n_tgt,  n_nodes, 1)
            intervention_indices: (batch,) int64
            target_train_data:    optional (batch, n_tgt, n_nodes, 1)
            variable_mask:        optional
        Returns:
            representation: (batch, n_ctx+n_tgt, n_nodes, d_model)
            num_target: int
        """
        context_embed = self.embed(context_data, intervention_indices, context=True)
        target_embed  = self.embed(target_data,  intervention_indices, context=False)

        target_train_embed = None
        if target_train_data is not None:
            target_train_embed = self.embed(target_train_data, intervention_indices, context=False)

        full_data   = self.create_input(context_embed, target_embed, target_train_embed)
        sample_mask, num_target = self.create_sample_mask(context_embed, target_embed, target_train_embed)

        representation = self.encoder(
            full_data,
            sample_mask=sample_mask,
            num_target=num_target,
            variable_mask=variable_mask,
        )
        return representation, num_target


class JointMoGCausalInferenceModel(BaseCausalInference):
    """
    MACE encoder + joint multivariate GMM decoder.

    Inherits BaseCausalInference to reuse the MACE Transformer encoder.
    Replaces the per-variable MoG predictor with MultivariateMoGPredictor
    which outputs a single GMM over the full n_vars-dimensional space.

    Args:
        config: CausalInfModelConfig (d_model, nhead, num_layers_encoder,
                num_nodes, num_mixture_components, etc.)
    """

    def __init__(self, config: CausalInfModelConfig):
        super().__init__(config)
        # Replace the 3-role encoder (variable/intervention/outcome) with a
        # 2-role encoder (variable/intervention) — no designated outcome index needed.
        self.encoder = OutcomeFreeCausalInfEncoder(
            d_model=config.d_model,
            dim_feedforward=config.dim_feedforward,
            nhead=config.nhead,
            num_layers=config.num_layers_encoder,
            num_nodes=config.num_nodes,
            device=config.device,
            dtype=config.dtype,
            emb_depth=config.emb_depth,
            dropout=config.dropout,
            sample_attn_mode=config.sample_attn_mode,
            linear_attention=config.linear_attention,
        )
        self.predictor = MultivariateMoGPredictor(
            d_model=config.d_model,
            n_vars=config.num_nodes,
            n_components=config.num_mixture_components,
            hidden_dim=config.dim_feedforward,
        )

    def forward(
        self,
        context_data: torch.Tensor,
        target_data: torch.Tensor,
        intervention_index: torch.Tensor,
        variable_mask=None,
    ) -> JointModelOutput:
        """
        Forward pass.

        Args:
            context_data:       (batch, n_ctx, n_vars)  — observational samples
            target_data:        (batch, n_tgt, n_vars)  — interventional samples
            intervention_index: (batch,) int64          — index of intervened variable
            variable_mask:      optional mask (passed to encoder)

        Returns:
            JointModelOutput
        """
        B, n_tgt, D = target_data.shape

        # Encoder expects 4D: (batch, samples, nodes, 1)
        context_4d = context_data.unsqueeze(-1)  # (B, n_ctx, n_vars, 1)
        target_4d  = target_data.unsqueeze(-1)   # (B, n_tgt, n_vars, 1)

        # Mask target so encoder only sees the intervention variable
        # Uses inherited BaseCausalInference._mask_target_data which expects 4D
        masked_target_4d = self._mask_target_data(target_4d, intervention_index)

        # Encode using outcome-free 2-role encoder
        # encode() returns (rep, num_trgt) where rep: (B, n_ctx+n_tgt, n_vars, d_model)
        rep, num_trgt = self.encoder.encode(
            context_4d,
            masked_target_4d,
            target_train_data=None,
            intervention_indices=intervention_index,
            variable_mask=variable_mask,
        )

        # Take target portion of representation
        target_rep = rep[:, -num_trgt:, :, :]   # (B, n_tgt, n_vars, d_model)

        # Pool over target samples to get per-node encodings
        node_encodings = target_rep.mean(dim=1)  # (B, n_vars, d_model)

        # Predict GMM parameters
        means, L, weights = self.predictor(node_encodings)

        return JointModelOutput(
            means=means,
            L=L,
            weights=weights,
            target_data=target_data,
        )

    def calculate_loss(
        self,
        model_output: JointModelOutput,
        target: torch.Tensor,
        test: bool = False,
    ) -> torch.Tensor:
        """
        Negative log-likelihood under the joint GMM.

        Args:
            model_output: JointModelOutput from forward()
            target:       (batch, n_tgt, D)
            test:         if True, return per-sample NLL; else return mean

        Returns:
            Scalar NLL (or tensor of shape (batch, n_tgt) when test=True)
        """
        # Cast to float32: linalg.solve_triangular (used in MultivariateNormal.log_prob)
        # has no bfloat16/float16 CUDA kernel, so the loss must run in float32.
        means   = model_output.means.float()    # (batch, K, D)
        L       = model_output.L.float()        # (batch, K, D, D)
        weights = model_output.weights.float()  # (batch, K)
        target  = target.float()

        # Unsqueeze sample dim so batch_shape=(batch,1) broadcasts with target (batch, n_tgt, D)
        mix  = Categorical(probs=weights.unsqueeze(1))           # (batch, 1, K)
        comp = MultivariateNormal(loc=means.unsqueeze(1),        # (batch, 1, K, D)
                                  scale_tril=L.unsqueeze(1))     # (batch, 1, K, D, D)
        gmm  = MixtureSameFamily(mix, comp)

        # target: (batch, n_tgt, D) — evaluate log_prob for each target sample
        # gmm.batch_shape=(batch,1) broadcasts → returns (batch, n_tgt)
        log_probs = gmm.log_prob(target)  # (batch, n_tgt)
        nll = -log_probs

        if test:
            return nll
        return nll.mean()

    @torch.no_grad()
    def predict(
        self,
        context_data: torch.Tensor,
        intervention_index: torch.Tensor,
        intervention_value: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample from the predicted joint GMM.

        Args:
            context_data:        (batch, n_ctx, n_vars) — raw observational samples
            intervention_index:  (batch,) int64 — which variable is intervened on
            intervention_value:  (batch,) float | None — the value to intervene with,
                                 in original (un-normalized) data space.
                                 If None, uses 0 in normalized space (population mean).
            n_samples:           number of samples to draw

        Returns:
            samples: (batch, n_samples, n_vars) in original data space
        """
        B, n_ctx, D = context_data.shape

        # Normalize context with the same z-score used during training
        mean = context_data.mean(dim=1, keepdim=True)  # (B, 1, D)
        std  = context_data.std(dim=1, keepdim=True) + 1e-8
        context_norm = (context_data - mean) / std

        # Build the target token: a single sample showing only the intervention value.
        # This tells the encoder "query the distribution under do(X_j = v)".
        dummy_target = torch.zeros(B, 1, D, device=context_data.device,
                                   dtype=context_data.dtype)
        if intervention_value is not None:
            # intervention_value is in original space; normalize it
            batch_idx = torch.arange(B, device=context_data.device)
            var_mean = mean[batch_idx, 0, intervention_index]  # (B,)
            var_std  = std[batch_idx, 0, intervention_index]   # (B,)
            v_norm   = (intervention_value - var_mean) / var_std
            dummy_target[batch_idx, 0, intervention_index] = v_norm

        out = self.forward(context_norm, dummy_target, intervention_index)

        means   = out.means    # (batch, K, D)
        L       = out.L        # (batch, K, D, D)
        weights = out.weights  # (batch, K)

        mix  = Categorical(probs=weights)
        comp = MultivariateNormal(loc=means, scale_tril=L)
        gmm  = MixtureSameFamily(mix, comp)

        # Sample in normalized space: (n_samples, batch, D) → (batch, n_samples, D)
        samples_norm = gmm.sample((n_samples,)).permute(1, 0, 2)

        # Denormalize back to original data scale
        samples = samples_norm * std + mean
        return samples
