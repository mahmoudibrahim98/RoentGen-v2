"""
Demographic Encoder for V4 Architecture

A lightweight encoder that represents demographics (age, sex, race) as learned
embeddings rather than text tokens. This avoids the token budget limitation of
CLIP and provides a clean separation between clinical and demographic conditioning.

Key differences from HCN (V2):
- Simpler: No hierarchical composition, just embeddings + MLP
- Cleaner: Demographics as categorical indices, not text
- Focused: Strong auxiliary supervision for forcing meaningful embeddings

Architecture:
    (age_idx, sex_idx, race_idx) → Embeddings → MLP → [B, 1, d_ctx]
    
Why this should work where V3 failed:
- No CLIP modification (keeps pre-trained positional embeddings)
- Concatenates after encoding, not before (no sequence length issues)
- Strong auxiliary losses force embeddings to be discriminative
- Optional dropout strategy forces UNet to use demographic encoder

Authors: RoentGen V4 Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DemographicEncoder(nn.Module):
    """
    Lightweight demographic encoder for V4 architecture.
    
    Encodes demographics as learned categorical embeddings rather than text.
    Provides strong auxiliary supervision to ensure embeddings are meaningful.
    
    Args:
        num_age_bins: Number of age categories (default: 5)
        num_sex: Number of sex categories (default: 2)
        num_race: Number of race categories (default: 4)
        d_hidden: Hidden dimension for embeddings (default: 256)
        d_output: Output dimension matching UNet cross_attention_dim (default: 1024)
        dropout: Dropout probability (default: 0.1)
        
    Input:
        age_idx: [B] Long tensor of age bin indices (0 to num_age_bins-1)
        sex_idx: [B] Long tensor of sex indices (0 to num_sex-1)
        race_idx: [B] Long tensor of race indices (0 to num_race-1)
        
    Output:
        demo_ctx: [B, 1, d_output] - Demographic context to concatenate with text
        aux_logits: Dict with 'age', 'sex', 'race' logits for auxiliary losses
    """
    
    def __init__(
        self,
        num_age_bins: int = 5,
        num_sex: int = 2,
        num_race: int = 4,
        d_hidden: int = 256,
        d_output: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Store config for saving/loading
        self.config = {
            'num_age_bins': num_age_bins,
            'num_sex': num_sex,
            'num_race': num_race,
            'd_hidden': d_hidden,
            'd_output': d_output,
            'dropout': dropout,
        }
        
        self.num_age_bins = num_age_bins
        self.num_sex = num_sex
        self.num_race = num_race
        self.d_hidden = d_hidden
        self.d_output = d_output
        
        # === Categorical Embeddings ===
        # These are learned from scratch (not from CLIP)
        self.emb_age = nn.Embedding(num_age_bins, d_hidden)
        self.emb_sex = nn.Embedding(num_sex, d_hidden)
        self.emb_race = nn.Embedding(num_race, d_hidden)
        
        # === Fusion MLP ===
        # Combines the three demographic embeddings into a single representation
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_hidden, d_hidden * 2),
            nn.LayerNorm(d_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_output),
        )
        
        # === Auxiliary Classification Heads ===
        # These provide strong supervision to force embeddings to be discriminative
        # Without these, the model might ignore the demographic encoder
        self.age_classifier = nn.Linear(d_output, num_age_bins)
        self.sex_classifier = nn.Linear(d_output, num_sex)
        self.race_classifier = nn.Linear(d_output, num_race)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings and linear layers with small values."""
        # Initialize embeddings with small values
        nn.init.normal_(self.emb_age.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_sex.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.emb_race.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        age_idx: torch.Tensor,
        sex_idx: torch.Tensor,
        race_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the demographic encoder.
        
        Args:
            age_idx: [B] Long tensor of age bin indices
            sex_idx: [B] Long tensor of sex indices  
            race_idx: [B] Long tensor of race indices
            
        Returns:
            demo_ctx: [B, 1, d_output] - Context token for UNet
            aux_logits: Dict with 'age', 'sex', 'race' logits for auxiliary losses
        """
        # Get embeddings for each demographic attribute
        age_emb = self.emb_age(age_idx)      # [B, d_hidden]
        sex_emb = self.emb_sex(sex_idx)      # [B, d_hidden]
        race_emb = self.emb_race(race_idx)   # [B, d_hidden]
        
        # Concatenate all embeddings
        h = torch.cat([age_emb, sex_emb, race_emb], dim=-1)  # [B, 3*d_hidden]
        
        # Fuse into a single demographic representation
        demo_repr = self.fusion(h)  # [B, d_output]
        
        # Add sequence dimension for concatenation with text tokens
        demo_ctx = demo_repr.unsqueeze(1)  # [B, 1, d_output]
        
        # Compute auxiliary logits for supervision
        # These force the embeddings to be discriminative
        aux_logits = {
            'age': self.age_classifier(demo_repr),    # [B, num_age_bins]
            'sex': self.sex_classifier(demo_repr),    # [B, num_sex]
            'race': self.race_classifier(demo_repr),  # [B, num_race]
        }
        
        return demo_ctx, aux_logits
    
    def save_pretrained(self, save_directory: str):
        """Save the demographic encoder config and weights."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save weights
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)
        
        print(f"DemographicEncoder saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load a demographic encoder from saved config and weights."""
        import os
        import json
        
        # Load config
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(**config)
        
        # Load weights
        weights_path = os.path.join(load_directory, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"DemographicEncoder loaded from {load_directory}")
        return model


def load_demographic_encoder(args, logger) -> Optional[DemographicEncoder]:
    """
    Load or create a DemographicEncoder based on config.
    
    Args:
        args: Config object with demographic encoder settings
        logger: Logger for status messages
        
    Returns:
        DemographicEncoder if args.use_demographic_encoder is True, else None
    """
    if not getattr(args, 'use_demographic_encoder', False):
        logger.info("DemographicEncoder disabled (use_demographic_encoder=False)")
        return None
    
    logger.info("=" * 60)
    logger.info("Loading DemographicEncoder (V4)")
    logger.info("=" * 60)
    
    # Get config parameters
    num_age_bins = getattr(args, 'demo_num_age_bins', 5)
    num_sex = getattr(args, 'demo_num_sex', 2)
    num_race = getattr(args, 'demo_num_race', 4)
    d_hidden = getattr(args, 'demo_d_hidden', 256)
    d_output = getattr(args, 'demo_d_output', 1024)
    dropout = getattr(args, 'demo_dropout', 0.1)
    
    # Create encoder
    demo_encoder = DemographicEncoder(
        num_age_bins=num_age_bins,
        num_sex=num_sex,
        num_race=num_race,
        d_hidden=d_hidden,
        d_output=d_output,
        dropout=dropout,
    )
    
    logger.info(f"  Num age bins: {num_age_bins}")
    logger.info(f"  Num sex categories: {num_sex}")
    logger.info(f"  Num race categories: {num_race}")
    logger.info(f"  Hidden dim: {d_hidden}")
    logger.info(f"  Output dim: {d_output}")
    logger.info(f"  Dropout: {dropout}")
    
    # Count parameters
    num_params = sum(p.numel() for p in demo_encoder.parameters())
    logger.info(f"  Total parameters: {num_params:,}")
    
    # Optional: Load from pretrained checkpoint
    pretrained_path = getattr(args, 'demographic_encoder_pretrained_path', None)
    if pretrained_path:
        logger.info(f"  Loading from pretrained: {pretrained_path}")
        demo_encoder = DemographicEncoder.from_pretrained(pretrained_path)
    
    logger.info("=" * 60)
    
    return demo_encoder







