"""
Hierarchical Conditioner Network (HCN) for Compositional Demographic Embeddings

This module implements a hierarchical approach to learning demographic representations
for fairness in medical image generation. It addresses the data scarcity problem at
demographic intersections by leveraging compositional structure.

Architecture:
    Grandparents (single attributes) → Parents (pairwise) → Child (triple) → Context

Key Features:
    - Compositional learning: Leverages abundant single-attribute data
    - Uncertainty quantification: Knows when it's uncertain about rare groups
    - Hierarchical composition: Age×Sex, Age×Race, Sex×Race → Age×Sex×Race

Authors: [Your names]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """
    Multi-layer perceptron with LayerNorm and SiLU activation.

    Args:
        d_in: Input dimension
        d_hidden: Hidden layer dimension
        d_out: Output dimension
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HierarchicalConditioner(nn.Module):
    """
    Hierarchical Conditioning Network for compositional demographic embeddings.

    Learns demographic representations at three levels:
    - Grandparents: Single attributes (age, sex, race)
    - Parents: Pairwise compositions (age×sex, age×race, sex×race)
    - Child: Triple composition (age×sex×race)

    The hierarchical structure allows the model to leverage abundant data from
    single attributes and pairs to improve representations of rare intersections.

    Args:
        num_age_bins: Number of age categories
        num_sex: Number of sex categories (typically 2: M/F)
        num_race: Number of race/ethnicity categories
        d_node: Hidden dimension for embeddings (default: 256)
        d_ctx: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        use_uncertainty: Whether to output mu/logsigma for variational sampling

    Input:
        age_idx: [B] Long tensor of age bin indices (0 to num_age_bins-1)
        sex_idx: [B] Long tensor of sex indices (0 to num_sex-1)
        race_idx: [B] Long tensor of race indices (0 to num_race-1)

    Output:
        ctx: [B, 1, d_ctx] - Demographic context token to concatenate with text
        mu: [B, d_node] - Mean of variational distribution
        logsigma: [B, d_node] - Log std of variational distribution
    """
    def __init__(
        self,
        num_age_bins: int,
        num_sex: int,
        num_race: int,
        d_node: int = 256,
        d_ctx: int = 1024,
        dropout: float = 0.1,
        use_uncertainty: bool = True,
    ):
        super().__init__()

        # Store config for saving/loading
        self.config = {
            'num_age_bins': num_age_bins,
            'num_sex': num_sex,
            'num_race': num_race,
            'd_node': d_node,
            'd_ctx': d_ctx,
            'dropout': dropout,
            'use_uncertainty': use_uncertainty,
        }

        self.num_age = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_node = d_node
        self.d_ctx = d_ctx
        self.use_uncertainty = use_uncertainty

        # === Grandparent embeddings (single attributes) ===
        self.emb_age = nn.Embedding(num_age_bins, d_node)
        self.emb_sex = nn.Embedding(num_sex, d_node)
        self.emb_race = nn.Embedding(num_race, d_node)

        # === Parent composers (pairwise compositions) ===
        # These learn how age and sex interact, age and race interact, etc.
        self.compose_age_sex = MLP(
            d_in=2 * d_node,
            d_hidden=2 * d_node,
            d_out=d_node,
            dropout=dropout
        )
        self.compose_age_race = MLP(
            d_in=2 * d_node,
            d_hidden=2 * d_node,
            d_out=d_node,
            dropout=dropout
        )
        self.compose_sex_race = MLP(
            d_in=2 * d_node,
            d_hidden=2 * d_node,
            d_out=d_node,
            dropout=dropout
        )

        # === Child composer (triple composition from all parents) ===
        self.compose_all = MLP(
            d_in=3 * d_node,  # Three parent embeddings
            d_hidden=2 * d_node,
            d_out=d_node,
            dropout=dropout
        )

        # === Uncertainty heads (for rare group detection) ===
        if use_uncertainty:
            self.mu_head = nn.Linear(d_node, d_node)
            self.logsigma_head = nn.Linear(d_node, d_node)

        # === Project to UNet cross-attention dimension ===
        self.proj_ctx = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, d_ctx),
        )

        # === Auxiliary demographic classifiers (for diagnostic losses) ===
        self.age_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_age_bins),
        )
        self.sex_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_sex),
        )
        self.race_classifier = nn.Sequential(
            nn.LayerNorm(d_node),
            nn.Linear(d_node, num_race),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small normal distribution."""
        for emb in [self.emb_age, self.emb_sex, self.emb_race]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        # Initialize uncertainty heads conservatively
        if self.use_uncertainty:
            nn.init.normal_(self.mu_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(self.logsigma_head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.logsigma_head.bias, -1.0)  # Start with low variance

    def forward(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through hierarchical conditioning network.

        Args:
            age_idx: [B] Long tensor of age bin indices
            sex_idx: [B] Long tensor of sex indices
            race_idx: [B] Long tensor of race indices

        Returns:
            ctx: [B, 1, d_ctx] - Demographic context to concatenate with text
            mu: [B, d_node] - Mean of variational distribution
            logsigma: [B, d_node] - Log std of variational distribution
        """
        # === Level 1: Grandparent embeddings (single attributes) ===
        e_age = self.emb_age(age_idx)    # [B, d_node]
        e_sex = self.emb_sex(sex_idx)    # [B, d_node]
        e_race = self.emb_race(race_idx) # [B, d_node]

        # === Level 2: Parent compositions (pairwise) ===
        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], dim=-1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], dim=-1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], dim=-1))

        # === Level 3: Child composition (from all parents) ===
        h_child = self.compose_all(
            torch.cat([h_age_sex, h_age_race, h_sex_race], dim=-1)
        )

        # === Uncertainty quantification (variational) ===
        if self.use_uncertainty:
            mu = self.mu_head(h_child)
            logsigma = torch.clamp(
                self.logsigma_head(h_child),
                min=-5.0,  # Minimum variance (stable training)
                max=1.0    # Maximum variance (prevent explosion)
            )

            # Sample during training (reparameterization trick)
            # Use mean during inference (deterministic)
            if self.training:
                z = mu + torch.exp(logsigma) * torch.randn_like(mu)
            else:
                z = mu
        else:
            mu = h_child
            logsigma = torch.zeros_like(h_child)
            z = h_child

        # === Project to context dimension and add sequence axis ===
        ctx = self.proj_ctx(z).unsqueeze(1)  # [B, 1, d_ctx]

        # === Auxiliary logits (use mu which is deterministic at inference) ===
        aux_logits = {
            "age": self.age_classifier(mu),
            "sex": self.sex_classifier(mu),
            "race": self.race_classifier(mu),
        }

        return ctx, mu, logsigma, aux_logits

    def compute_compositional_loss(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute compositional consistency loss.

        Enforces that the hierarchical composition is consistent with
        simple additive composition of grandparent embeddings.

        Intuition: E(age, sex, race) should be predictable from
        E(age) + E(sex) + E(race) in direction (not necessarily magnitude).

        Args:
            age_idx: [B] Age indices
            sex_idx: [B] Sex indices
            race_idx: [B] Race indices

        Returns:
            loss_comp: Scalar compositional loss (cosine-based)
        """
        # Get grandparent embeddings
        e_age = self.emb_age(age_idx)
        e_sex = self.emb_sex(sex_idx)
        e_race = self.emb_race(race_idx)

        # Hierarchical composition (normal forward pass)
        h_age_sex = self.compose_age_sex(torch.cat([e_age, e_sex], -1))
        h_age_race = self.compose_age_race(torch.cat([e_age, e_race], -1))
        h_sex_race = self.compose_sex_race(torch.cat([e_sex, e_race], -1))
        h_child = self.compose_all(torch.cat([h_age_sex, h_age_race, h_sex_race], -1))

        # Simple additive baseline (perfect compositionality would match this)
        h_additive = e_age + e_sex + e_race

        # Cosine similarity (direction alignment, not magnitude)
        # Loss = 1 - cos_sim, so 0 when perfectly aligned, 2 when opposite
        cos_sim = F.cosine_similarity(h_child, h_additive, dim=-1)
        loss_comp = (1 - cos_sim).mean()

        return loss_comp

    def get_uncertainty(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get uncertainty (sigma) for given demographic groups.
        Useful for detecting which groups the model is uncertain about.

        Returns:
            sigma: [B] - Standard deviation for each sample
        """
        _, _, logsigma = self.forward(age_idx, sex_idx, race_idx)
        sigma = torch.exp(logsigma).mean(dim=-1)  # Average across dimensions
        return sigma

    def save_pretrained(self, save_dir: str):
        """Save HCN model and config."""
        import os
        import json

        os.makedirs(save_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Save weights
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        print(f"HCN saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu"):
        """Load HCN model from saved checkpoint."""
        import os
        import json

        # Load config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model
        model = cls(**config)

        # Load weights
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print(f"HCN loaded from {save_dir}")
        return model


def test_hcn():
    """Quick test of HCN module."""
    print("Testing HCN module...")

    # Create HCN
    hcn = HierarchicalConditioner(
        num_age_bins=5,
        num_sex=2,
        num_race=4,
        d_node=256,
        d_ctx=1024,
    )

    # Test forward pass
    batch_size = 8
    age = torch.randint(0, 5, (batch_size,))
    sex = torch.randint(0, 2, (batch_size,))
    race = torch.randint(0, 4, (batch_size,))

    hcn.train()
    ctx, mu, logsigma, aux_logits = hcn(age, sex, race)

    assert ctx.shape == (batch_size, 1, 1024), f"Expected (8, 1, 1024), got {ctx.shape}"
    assert mu.shape == (batch_size, 256), f"Expected (8, 256), got {mu.shape}"
    assert logsigma.shape == (batch_size, 256), f"Expected (8, 256), got {logsigma.shape}"
    assert all(k in aux_logits for k in ("age", "sex", "race"))

    print(f"✓ Forward pass: ctx shape = {ctx.shape}")

    # Test compositional loss
    comp_loss = hcn.compute_compositional_loss(age, sex, race)
    assert comp_loss.ndim == 0, "Compositional loss should be scalar"
    print(f"✓ Compositional loss: {comp_loss.item():.4f}")

    # Test uncertainty
    sigma = hcn.get_uncertainty(age, sex, race)
    assert sigma.shape == (batch_size,), f"Expected ({batch_size},), got {sigma.shape}"
    print(f"✓ Uncertainty: mean sigma = {sigma.mean().item():.4f}")

    # Test save/load
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    try:
        hcn.save_pretrained(temp_dir)
        hcn_loaded = HierarchicalConditioner.from_pretrained(temp_dir)
        ctx_loaded, _, _, _ = hcn_loaded(age, sex, race)
        print(f"✓ Save/load successful")
    finally:
        shutil.rmtree(temp_dir)

    print(f"✓ All tests passed!")
    print(f"✓ Total parameters: {sum(p.numel() for p in hcn.parameters()):,}")


if __name__ == "__main__":
    test_hcn()
