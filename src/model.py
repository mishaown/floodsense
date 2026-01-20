"""
FloodSense: Lightweight Bi-Temporal Flood Detection Model

Components:
- Shared CNN encoder (Siamese)
- ConvLSTM for temporal sequence modeling
- Temporal attention for change-aware fusion
- Multi-scale difference aggregation
- Lightweight decoder for segmentation

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
    in_channels: int = 3
    num_classes: int = 2
    encoder: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 1
    use_attention: bool = True
    dropout: float = 0.1
    img_size: int = 256


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=True)

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.channels = channels
        mid_ch = max(channels // reduction, 16)
        self.query = nn.Conv2d(channels, mid_ch, 1)
        self.key = nn.Conv2d(channels, mid_ch, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.diff_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels, 1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_t0.shape
        Q = self.query(feat_t0).view(B, -1, H * W)
        K = self.key(feat_t1).view(B, -1, H * W)
        V = self.value(feat_t1).view(B, C, H * W)
        attn = torch.bmm(Q.transpose(1, 2), K)
        attn = F.softmax(attn / (C ** 0.5), dim=-1)
        attended = torch.bmm(V, attn.transpose(1, 2))
        attended = attended.view(B, C, H, W)
        diff = torch.abs(feat_t1 - feat_t0)
        diff_weight = self.diff_attn(diff)
        weighted_diff = diff * diff_weight
        change_feat = feat_t0 + self.gamma * attended + weighted_diff
        return change_feat


class DifferenceModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.diff_conv = nn.Sequential(
            ConvBlock(in_channels * 2, out_channels),
            ConvBlock(out_channels, out_channels)
        )
        self.abs_diff_conv = nn.Sequential(
            ConvBlock(in_channels, out_channels // 2),
        )
        self.fusion = nn.Sequential(
            ConvBlock(out_channels + out_channels // 2, out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, feat_t0: torch.Tensor, feat_t1: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([feat_t0, feat_t1], dim=1)
        concat_diff = self.diff_conv(concat)
        abs_diff = torch.abs(feat_t1 - feat_t0)
        abs_diff = self.abs_diff_conv(abs_diff)
        fused = torch.cat([concat_diff, abs_diff], dim=1)
        return self.fusion(fused)


class LightweightDecoder(nn.Module):
    def __init__(self, encoder_channels: List[int], hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.laterals = nn.ModuleList([nn.Conv2d(ch, hidden_dim, 1) for ch in encoder_channels])
        self.decoder_blocks = nn.ModuleList([ConvBlock(hidden_dim, hidden_dim) for _ in range(len(encoder_channels))])
        self.classifier = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        x = None
        for i in range(len(features) - 1, -1, -1):
            lateral = self.laterals[i](features[i])
            if x is not None:
                x = F.interpolate(x, size=lateral.shape[-2:], mode='bilinear', align_corners=False)
                x = x + lateral
            else:
                x = lateral
            x = self.decoder_blocks[i](x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        logits = self.classifier(x)
        return logits


class FloodSense(nn.Module):
    def __init__(self, config: Union[FloodSenseModelConfig, dict]):
        super().__init__()
        if isinstance(config, dict):
            self.config = FloodSenseModelConfig(**{k: v for k, v in config.items() if k in FloodSenseModelConfig.__dataclass_fields__})
        else:
            self.config = config

        self.encoder = self._build_encoder()
        encoder_channels = self._get_encoder_channels()

        self.convlstm = ConvLSTMCell(input_dim=encoder_channels[-1], hidden_dim=self.config.lstm_hidden)

        if self.config.use_attention:
            self.temporal_attn = TemporalAttention(channels=encoder_channels[-1], reduction=4)
        else:
            self.temporal_attn = None

        self.diff_modules = nn.ModuleList([DifferenceModule(ch, self.config.hidden_dim) for ch in encoder_channels])
        self._lstm_proj = nn.Conv2d(self.config.lstm_hidden, self.config.hidden_dim, 1)

        self.fusion = nn.Sequential(
            ConvBlock(self.config.hidden_dim * len(encoder_channels), self.config.hidden_dim * 2),
            ConvBlock(self.config.hidden_dim * 2, self.config.hidden_dim),
        )

        self.decoder = LightweightDecoder(
            encoder_channels=[self.config.hidden_dim] * len(encoder_channels),
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes
        )

        self.magnitude_head = nn.Sequential(
            ConvBlock(self.config.hidden_dim, 32),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def _build_encoder(self) -> nn.Module:
        encoder_name = self.config.encoder
        if encoder_name.startswith('resnet'):
            encoder = timm.create_model(encoder_name, pretrained=self.config.pretrained, features_only=True, out_indices=[1, 2, 3, 4], in_chans=self.config.in_channels)
        elif encoder_name.startswith('efficientnet'):
            encoder = timm.create_model(encoder_name, pretrained=self.config.pretrained, features_only=True, out_indices=[1, 2, 3, 4], in_chans=self.config.in_channels)
        else:
            encoder = timm.create_model('resnet50', pretrained=self.config.pretrained, features_only=True, out_indices=[1, 2, 3, 4], in_chans=self.config.in_channels)
        return encoder

    def _get_encoder_channels(self) -> List[int]:
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels, 64, 64)
            features = self.encoder(dummy)
            channels = [f.shape[1] for f in features]
        return channels

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    def forward(self, img_t0: torch.Tensor, img_t1: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        B, C, H, W = img_t0.shape
        feats_t0 = self.encode(img_t0)
        feats_t1 = self.encode(img_t1)

        diff_feats = []
        for i, (f0, f1) in enumerate(zip(feats_t0, feats_t1)):
            diff = self.diff_modules[i](f0, f1)
            diff_feats.append(diff)

        if self.temporal_attn is not None:
            deep_attn = self.temporal_attn(feats_t0[-1], feats_t1[-1])
            lstm_out, _ = self.convlstm(feats_t0[-1], None)
            lstm_out, _ = self.convlstm(feats_t1[-1], (lstm_out, lstm_out))

        target_size = diff_feats[0].shape[-2:]
        upsampled = []
        for feat in diff_feats:
            up = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(up)

        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)

        decoder_feats = []
        for i, diff in enumerate(diff_feats):
            if i == len(diff_feats) - 1 and self.temporal_attn is not None:
                lstm_proj = self._lstm_proj(lstm_out)
                lstm_up = F.interpolate(lstm_proj, size=diff.shape[-2:], mode='bilinear', align_corners=False)
                diff = diff + lstm_up
            decoder_feats.append(diff)

        logits = self.decoder(decoder_feats, (H, W))
        magnitude = self.magnitude_head(fused)
        magnitude = F.interpolate(magnitude, size=(H, W), mode='bilinear', align_corners=False)

        outputs = {'logits': logits, 'magnitude': magnitude}
        if return_features:
            outputs['features'] = {'feats_t0': feats_t0, 'feats_t1': feats_t1, 'diff_feats': diff_feats, 'fused': fused}
        return outputs

    def predict(self, img_t0: torch.Tensor, img_t1: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(img_t0, img_t1)
            probs = F.softmax(outputs['logits'], dim=1)
            preds = probs.argmax(dim=1)
            return {'predictions': preds, 'probabilities': probs, 'magnitude': outputs['magnitude']}

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_model(config: Union[FloodSenseModelConfig, dict]) -> FloodSense:
    return FloodSense(config)
