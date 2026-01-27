"""
FloodSense: Theoretically Correct Bi-Temporal Flood Detection Architecture

Architecture Flow (temporal_mode="all"):
    TFEN (4 scales) → CTAM (all scales) → STSM (all scales) → MSDAM (all scales) → PUD → Flood Map
                                                                                  ↓
                                                                                HFFM
                                                                                  ↓
                                                                                CMH → Magnitude Map

Architecture Flow (temporal_mode="deepest"):
    TFEN (4 scales) → [Scale 1-3: direct] ────────────────────→ MSDAM (all scales) → PUD → Flood Map
                    → [Scale 4: CTAM → STSM] ─────────────────↗                    ↓
                                                                                  HFFM
                                                                                    ↓
                                                                                  CMH → Magnitude Map

Configuration:
    temporal_mode: "all" (default) - CTAM+STSM for all 4 scales
                   "deepest" - CTAM+STSM only for deepest layer (computationally efficient)

Components:
    - TFEN: Temporal Feature Extraction Network (Siamese encoder)
    - CTAM: Cross-Temporal Attention Module
    - STSM: Spatial-Temporal Sequence Module
    - MSDAM: Multi-Scale Difference Aggregation Module
    - PUD: Progressive Upsampling Decoder
    - HFFM: Hierarchical Feature Fusion Module
    - CMH: Change Magnitude Head

Input: (pre_flood, post_flood) image pairs
Output: {'logits': flood_mask, 'magnitude': change_intensity}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List, Tuple
from dataclasses import dataclass
import timm


@dataclass
class FloodSenseModelConfig:
    """Configuration for FloodSense model."""
    in_channels: int = 3
    num_classes: int = 2
    encoder: str = "efficientnetv2_rw_t"
    pretrained: bool = True
    hidden_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 1
    use_attention: bool = True
    dropout: float = 0.1
    img_size: int = 256
    attention_reduction: int = 4
    temporal_mode: str = "all"  # "all" = all scales, "deepest" = only deepest layer


class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell for spatial-temporal modeling.

    Preserves spatial structure through convolutional gates rather than
    fully-connected operations, essential for dense prediction tasks.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # Single convolution for all four gates (input, forget, output, cell)
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
            bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through ConvLSTM cell.

        Args:
            x: Input tensor of shape (B, C, H, W)
            state: Tuple of (hidden_state, cell_state) or None for initialization

        Returns:
            h_new: New hidden state (B, hidden_dim, H, W)
            (h_new, c_new): Tuple of new hidden and cell states
        """
        B, _, H, W = x.shape

        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into four gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate

        # Update cell and hidden states
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, (h_new, c_new)


class CrossTemporalAttentionModule(nn.Module):
    """
    Cross-Temporal Attention Module (CTAM).

    Computes change-aware features through:
    1. Cross-temporal spatial attention (query from t0, key-value from t1)
    2. Channel difference attention for adaptive re-weighting
    3. Learnable gamma parameter for stable training

    Complexity: O(C²/r) where r is the reduction ratio
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels // reduction, 16)

        # Cross-temporal attention projections
        self.query = nn.Conv2d(channels, mid_ch, 1)
        self.key = nn.Conv2d(channels, mid_ch, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # Channel difference attention
        self.diff_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1),
            nn.Sigmoid()
        )

        # Learnable fusion parameter (initialized at zero for training stability)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-temporal attention.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Change-aware features (B, C, H, W)
        """
        B, C, H, W = feat_t0.shape

        # Cross-temporal spatial attention
        Q = self.query(feat_t0).view(B, -1, H * W)      # Query from pre-flood
        K = self.key(feat_t1).view(B, -1, H * W)        # Key from post-flood
        V = self.value(feat_t1).view(B, C, H * W)       # Value from post-flood

        # Compute attention weights
        attn = torch.bmm(Q.transpose(1, 2), K)
        attn = F.softmax(attn / (C ** 0.5), dim=-1)

        # Apply attention to values
        attended = torch.bmm(V, attn.transpose(1, 2))
        attended = attended.view(B, C, H, W)

        # Channel difference attention
        diff = torch.abs(feat_t1 - feat_t0)
        diff_weight = self.diff_attn(diff)
        weighted_diff = diff * diff_weight

        # Fuse with learnable gamma
        change_feat = feat_t0 + self.gamma * attended + weighted_diff

        return change_feat


class SpatialTemporalSequenceModule(nn.Module):
    """
    Spatial-Temporal Sequence Module (STSM).

    Wraps ConvLSTM for sequential bi-temporal processing.
    Processes pre-flood features first, then post-flood features
    using the propagated hidden and cell states.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.convlstm = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Sequential temporal processing: t0 → t1.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Temporally-modeled features (B, hidden_dim, H, W)
        """
        # Process pre-flood features (initialize states)
        h1, (h1_out, c1_out) = self.convlstm(feat_t0, None)

        # Process post-flood features with propagated states
        h2, (h2_out, c2_out) = self.convlstm(feat_t1, (h1_out, c1_out))

        return h2


class MultiScaleDifferenceAggregationModule(nn.Module):
    """
    Multi-Scale Difference Aggregation Module (MSDAM).

    Dual-path architecture:
    1. Concatenation path: Learns complex feature interactions
    2. Absolute difference path: Captures magnitude-based changes

    Asymmetric channel allocation (2:1 ratio) balances interaction
    capacity with explicit difference preservation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Concatenation path (processes joint features)
        self.concat_conv = nn.Sequential(
            ConvBlock(in_channels * 2, out_channels),
            ConvBlock(out_channels, out_channels)
        )

        # Absolute difference path
        self.diff_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2),
        )

        # Feature fusion (2:1 asymmetric ratio)
        self.fusion = nn.Sequential(
            ConvBlock(out_channels + out_channels // 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale difference features.

        Args:
            feat_t0: Pre-flood features (B, C, H, W)
            feat_t1: Post-flood features (B, C, H, W)

        Returns:
            Difference-aggregated features (B, out_channels, H, W)
        """
        # Concatenation path
        concat = torch.cat([feat_t0, feat_t1], dim=1)
        concat_feats = self.concat_conv(concat)

        # Absolute difference path
        abs_diff = torch.abs(feat_t1 - feat_t0)
        diff_feats = self.diff_conv(abs_diff)

        # Fuse with asymmetric channel allocation
        fused = torch.cat([concat_feats, diff_feats], dim=1)

        return self.fusion(fused)


class ProgressiveUpsamplingDecoder(nn.Module):
    """
    Progressive Upsampling Decoder (PUD).

    Lightweight FPN-style decoder with:
    1. Additive skip connections (parameter-efficient)
    2. Progressive coarse-to-fine refinement
    3. Unified channel dimension across all scales
    """

    def __init__(
        self,
        encoder_channels: List[int],
        hidden_dim: int = 64,
        num_classes: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Lateral connections (project to unified channel dimension)
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, hidden_dim, 1) for ch in encoder_channels
        ])

        # Decoder blocks for progressive refinement
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim) for _ in range(len(encoder_channels))
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Progressive decoding from coarse to fine.

        Args:
            features: List of multi-scale features [scale1, scale2, scale3, scale4]
            target_size: Output spatial dimensions (H, W)

        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        x = None

        # Process from coarsest to finest (reverse order)
        for i in range(len(features) - 1, -1, -1):
            lateral = self.laterals[i](features[i])

            if x is not None:
                # Upsample and add (additive skip connection)
                x = F.interpolate(
                    x, size=lateral.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                x = x + lateral
            else:
                x = lateral

            x = self.decoder_blocks[i](x)

        # Upsample to target resolution
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return self.classifier(x)


class HierarchicalFeatureFusionModule(nn.Module):
    """
    Hierarchical Feature Fusion Module (HFFM).

    Aggregates multi-scale features from different hierarchical levels
    for auxiliary magnitude prediction. The module performs:

    1. Scale normalization: Upsamples all features to a common resolution
    2. Channel concatenation: Combines features along channel dimension
    3. Feature fusion: Reduces concatenated features through conv layers

    This hierarchical fusion captures both fine-grained spatial details
    from shallow layers and semantic context from deeper layers,
    providing rich representations for change magnitude estimation.
    """

    def __init__(
        self,
        in_channels_per_scale: int,
        num_scales: int,
        hidden_dim: int = 64
    ):
        """
        Initialize HFFM.

        Args:
            in_channels_per_scale: Number of channels per scale (assumed uniform)
            num_scales: Number of hierarchical scales to fuse
            hidden_dim: Output channel dimension after fusion
        """
        super().__init__()

        self.num_scales = num_scales
        total_channels = in_channels_per_scale * num_scales

        # Two-stage fusion for gradual channel reduction
        self.fusion = nn.Sequential(
            ConvBlock(total_channels, hidden_dim * 2),
            ConvBlock(hidden_dim * 2, hidden_dim),
        )

    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse hierarchical features from multiple scales.

        Args:
            multi_scale_features: List of features from different scales
                                  [(B, C, H1, W1), (B, C, H2, W2), ...]
                                  Ordered from finest to coarsest resolution

        Returns:
            Fused features (B, hidden_dim, H1, W1) at finest input resolution
        """
        # Target resolution is the finest scale (first in list)
        target_size = multi_scale_features[0].shape[-2:]

        # Upsample all features to common resolution
        upsampled_features = []
        for feat in multi_scale_features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size,
                    mode='bilinear', align_corners=False
                )
            upsampled_features.append(feat)

        # Concatenate along channel dimension
        concatenated = torch.cat(upsampled_features, dim=1)

        # Apply fusion convolutions
        fused = self.fusion(concatenated)

        return fused


class FloodSense(nn.Module):
    """
    FloodSense: Theoretically Correct Bi-Temporal Flood Detection Model.

    Architecture:
        1. TFEN: Shared encoder extracts 4-scale feature pyramids from both images
        2. CTAM: Cross-temporal attention (all scales or deepest only)
        3. STSM: Temporal sequence modeling on CTAM-enhanced features
        4. MSDAM: Difference aggregation on temporally-modeled features
        5. PUD: Progressive decoder for flood segmentation
        6. HFFM: Hierarchical feature fusion for magnitude estimation
        7. CMH: Auxiliary magnitude head for enhanced supervision

    Temporal Modes:
        temporal_mode="all" (default):
            All 4 scales processed through CTAM → STSM → MSDAM
            Higher computational cost, richer temporal modeling

        temporal_mode="deepest":
            Only deepest layer: CTAM → STSM
            Scales 1-3: Direct to MSDAM (bypass CTAM/STSM)
            Lower computational cost, efficient for resource-constrained settings

    Data Flow (temporal_mode="all"):
        Pre/Post Images → TFEN → [4 scales]
                                    ↓
                            CTAM (per scale)
                                    ↓
                            STSM (per scale)
                                    ↓
                            MSDAM (per scale)
                                    ↓
                         ┌─────────┴─────────┐
                         ↓                   ↓
                        PUD                HFFM
                         ↓                   ↓
                    Flood Logits           CMH
                                            ↓
                                     Magnitude Map

    Data Flow (temporal_mode="deepest"):
        Pre/Post Images → TFEN → [Scales 1-3] ──────────────→ MSDAM
                               → [Scale 4] → CTAM → STSM ──→ MSDAM
                                                              ↓
                                                    ┌────────┴────────┐
                                                    ↓                 ↓
                                                   PUD              HFFM
                                                    ↓                 ↓
                                               Flood Logits         CMH
                                                                     ↓
                                                              Magnitude Map
    """

    def __init__(self, config: Union[FloodSenseModelConfig, dict]):
        super().__init__()

        if isinstance(config, dict):
            self.config = FloodSenseModelConfig(**{
                k: v for k, v in config.items()
                if k in FloodSenseModelConfig.__dataclass_fields__
            })
        else:
            self.config = config

        # Build shared encoder (TFEN)
        self.encoder = self._build_encoder()
        encoder_channels = self._get_encoder_channels()
        self.encoder_channels = encoder_channels
        self.temporal_mode = self.config.temporal_mode

        # CTAM and STSM modules based on temporal_mode
        if self.temporal_mode == "deepest":
            # Only deepest layer gets CTAM + STSM
            if self.config.use_attention:
                self.ctam_deepest = CrossTemporalAttentionModule(
                    channels=encoder_channels[-1],
                    reduction=self.config.attention_reduction
                )
            else:
                self.ctam_deepest = None

            self.stsm_deepest = SpatialTemporalSequenceModule(
                input_dim=encoder_channels[-1],
                hidden_dim=self.config.lstm_hidden
            )

            self.stsm_projection_deepest = nn.Conv2d(
                self.config.lstm_hidden, encoder_channels[-1], 1
            )

            # Set module lists to None for deepest mode
            self.ctam_modules = None
            self.stsm_modules = None
            self.stsm_projections = None

        else:  # "all" mode - CTAM + STSM for all scales
            if self.config.use_attention:
                self.ctam_modules = nn.ModuleList([
                    CrossTemporalAttentionModule(
                        channels=ch,
                        reduction=self.config.attention_reduction
                    ) for ch in encoder_channels
                ])
            else:
                self.ctam_modules = None

            self.stsm_modules = nn.ModuleList([
                SpatialTemporalSequenceModule(
                    input_dim=ch,
                    hidden_dim=self.config.lstm_hidden
                ) for ch in encoder_channels
            ])

            self.stsm_projections = nn.ModuleList([
                nn.Conv2d(self.config.lstm_hidden, ch, 1) for ch in encoder_channels
            ])

            # Set deepest-only modules to None
            self.ctam_deepest = None
            self.stsm_deepest = None
            self.stsm_projection_deepest = None

        # MSDAM modules for all scales
        self.msdam_modules = nn.ModuleList([
            MultiScaleDifferenceAggregationModule(
                in_channels=ch,
                out_channels=self.config.hidden_dim
            ) for ch in encoder_channels
        ])

        # Progressive Upsampling Decoder (PUD)
        self.decoder = ProgressiveUpsamplingDecoder(
            encoder_channels=[self.config.hidden_dim] * len(encoder_channels),
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes
        )

        # Hierarchical Feature Fusion Module (HFFM)
        # Aggregates multi-scale MSDAM features for magnitude estimation
        self.hffm = HierarchicalFeatureFusionModule(
            in_channels_per_scale=self.config.hidden_dim,
            num_scales=len(encoder_channels),
            hidden_dim=self.config.hidden_dim
        )

        # Change Magnitude Head (CMH)
        # Receives HFFM-fused features for auxiliary supervision
        self.magnitude_head = nn.Sequential(
            ConvBlock(self.config.hidden_dim, 32),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Dropout for regularization
        self.dropout = nn.Dropout2d(self.config.dropout)

    def _build_encoder(self) -> nn.Module:
        """Build the TFEN (Temporal Feature Extraction Network)."""
        encoder_name = self.config.encoder

        if encoder_name.startswith('resnet'):
            encoder = timm.create_model(
                encoder_name,
                pretrained=self.config.pretrained,
                features_only=True,
                out_indices=[1, 2, 3, 4],
                in_chans=self.config.in_channels
            )
        elif encoder_name.startswith('efficientnet'):
            encoder = timm.create_model(
                encoder_name,
                pretrained=self.config.pretrained,
                features_only=True,
                out_indices=[1, 2, 3, 4],
                in_chans=self.config.in_channels
            )
        else:
            # Default to ResNet50
            encoder = timm.create_model(
                'resnet50',
                pretrained=self.config.pretrained,
                features_only=True,
                out_indices=[1, 2, 3, 4],
                in_chans=self.config.in_channels
            )

        return encoder

    def _get_encoder_channels(self) -> List[int]:
        """Get channel dimensions at each encoder scale."""
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, 64, 64)
            features = self.encoder(dummy)
            channels = [f.shape[1] for f in features]
        return channels

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features using TFEN."""
        return self.encoder(x)

    def forward(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FloodSense.

        Args:
            img_t0: Pre-flood image (B, C, H, W)
            img_t1: Post-flood image (B, C, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - 'logits': Flood segmentation logits (B, num_classes, H, W)
                - 'magnitude': Change magnitude map (B, 1, H, W)
                - 'features': Intermediate features (if return_features=True)
        """
        B, C, H, W = img_t0.shape

        # Stage 1: TFEN - Extract multi-scale features
        feats_t0 = self.encode(img_t0)  # List of 4 scales
        feats_t1 = self.encode(img_t1)  # List of 4 scales

        # Stage 2 & 3: CTAM + STSM processing
        if self.temporal_mode == "deepest":
            # Only deepest layer goes through CTAM + STSM
            temporal_feats_t0 = []
            temporal_feats_t1 = []

            for i, (f0, f1) in enumerate(zip(feats_t0, feats_t1)):
                if i == len(feats_t0) - 1:  # Deepest layer
                    # Apply CTAM if enabled
                    if self.ctam_deepest is not None:
                        f0_attn = self.ctam_deepest(f0, f1)
                        f1_attn = self.ctam_deepest(f1, f0)  # Bidirectional
                    else:
                        f0_attn = f0
                        f1_attn = f1

                    # Apply STSM (receives CTAM-enhanced features)
                    stsm_out = self.stsm_deepest(f0_attn, f1_attn)

                    # Project STSM output back to encoder channel dimension
                    stsm_proj = self.stsm_projection_deepest(stsm_out)

                    # Combine with residual connection
                    temporal_feats_t0.append(f0_attn + stsm_proj)
                    temporal_feats_t1.append(f1_attn + stsm_proj)
                else:
                    # Shallow layers pass through directly (no CTAM/STSM)
                    temporal_feats_t0.append(f0)
                    temporal_feats_t1.append(f1)

        else:  # "all" mode - CTAM + STSM for all scales
            temporal_feats_t0 = []
            temporal_feats_t1 = []

            for i, (f0, f1) in enumerate(zip(feats_t0, feats_t1)):
                # Apply CTAM if enabled
                if self.ctam_modules is not None:
                    f0_attn = self.ctam_modules[i](f0, f1)
                    f1_attn = self.ctam_modules[i](f1, f0)  # Bidirectional attention
                else:
                    f0_attn = f0
                    f1_attn = f1

                # Apply STSM (receives CTAM-enhanced features)
                stsm_out = self.stsm_modules[i](f0_attn, f1_attn)

                # Project STSM output back to encoder channel dimension
                stsm_proj = self.stsm_projections[i](stsm_out)

                # Combine CTAM features with STSM temporal modeling
                # Add residual connection for gradient flow
                temporal_feats_t0.append(f0_attn + stsm_proj)
                temporal_feats_t1.append(f1_attn + stsm_proj)

        # Stage 4: MSDAM - Difference aggregation on temporally-modeled features
        msdam_feats = []
        for i, (f0, f1) in enumerate(zip(temporal_feats_t0, temporal_feats_t1)):
            diff_feat = self.msdam_modules[i](f0, f1)
            diff_feat = self.dropout(diff_feat)
            msdam_feats.append(diff_feat)

        # Stage 5: PUD - Progressive decoding for flood segmentation
        logits = self.decoder(msdam_feats, (H, W))

        # Stage 6: HFFM - Hierarchical feature fusion for magnitude estimation
        # Upsamples and concatenates multi-scale MSDAM features
        fused = self.hffm(msdam_feats)

        # Stage 7: CMH - Change magnitude prediction from HFFM output
        magnitude = self.magnitude_head(fused)
        magnitude = F.interpolate(
            magnitude, size=(H, W),
            mode='bilinear', align_corners=False
        )

        outputs = {
            'logits': logits,
            'magnitude': magnitude
        }

        if return_features:
            outputs['features'] = {
                'feats_t0': feats_t0,
                'feats_t1': feats_t1,
                'temporal_feats_t0': temporal_feats_t0,
                'temporal_feats_t1': temporal_feats_t1,
                'msdam_feats': msdam_feats,
                'hffm_output': fused
            }

        return outputs

    def predict(
        self,
        img_t0: torch.Tensor,
        img_t1: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode prediction.

        Args:
            img_t0: Pre-flood image (B, C, H, W)
            img_t1: Post-flood image (B, C, H, W)

        Returns:
            Dictionary containing predictions, probabilities, and magnitude
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img_t0, img_t1)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)

            return {
                'predictions': preds,
                'probabilities': probs,
                'magnitude': outputs['magnitude']
            }

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_module_parameters(self) -> Dict[str, int]:
        """Get parameter count per module for analysis."""
        counts = {
            'encoder (TFEN)': sum(p.numel() for p in self.encoder.parameters()),
            'msdam': sum(p.numel() for p in self.msdam_modules.parameters()),
            'decoder (PUD)': sum(p.numel() for p in self.decoder.parameters()),
            'hffm': sum(p.numel() for p in self.hffm.parameters()),
            'magnitude_head (CMH)': sum(p.numel() for p in self.magnitude_head.parameters()),
        }

        # Handle CTAM and STSM based on temporal mode
        if self.temporal_mode == "deepest":
            counts['ctam'] = sum(p.numel() for p in self.ctam_deepest.parameters()) if self.ctam_deepest else 0
            counts['stsm'] = sum(p.numel() for p in self.stsm_deepest.parameters())
            counts['stsm_proj'] = sum(p.numel() for p in self.stsm_projection_deepest.parameters())
        else:
            counts['ctam'] = sum(p.numel() for p in self.ctam_modules.parameters()) if self.ctam_modules else 0
            counts['stsm'] = sum(p.numel() for p in self.stsm_modules.parameters())
            counts['stsm_proj'] = sum(p.numel() for p in self.stsm_projections.parameters())

        counts['total'] = sum(counts.values())
        return counts


def build_model(config: Union[FloodSenseModelConfig, dict]) -> FloodSense:
    """Factory function to build FloodSense model."""
    return FloodSense(config)
