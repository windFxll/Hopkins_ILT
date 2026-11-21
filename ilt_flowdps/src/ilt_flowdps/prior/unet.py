"""U-Net definition for the ILT film5_bottleneck_mhsa32_v2 checkpoint."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _group_norm(channels):
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=time.device, dtype=torch.float32) / half
        ).to(dtype=time.dtype)
        phase = (time * 1000.0)[:, None] * frequency[None, :]
        return torch.cat((phase.cos(), phase.sin()), dim=1)


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.in_norm = _group_norm(in_channels)
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * out_channels))
        self.out_norm = _group_norm(out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, time_embedding):
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.time_projection(time_embedding).chunk(2, dim=1)
        h = self.out_norm(h) * (1.0 + scale[:, :, None, None])
        h = h + shift[:, :, None, None]
        return self.skip(x) + self.out_conv(F.silu(h))


class TimeConditionedMHSA(nn.Module):
    def __init__(self, channels, time_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_channels = channels // num_heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x, time_embedding):
        batch, channels, height, width = x.shape
        length = height * width
        residual = x.reshape(batch, channels, length)
        normalized = self.norm(x).reshape(batch, channels, length)
        qkv = self.qkv(normalized).reshape(batch, self.num_heads, 3 * self.head_channels, length)
        query, key, value = qkv.split(self.head_channels, dim=2)
        scale = self.head_channels ** -0.25
        attention = torch.einsum("bhct,bhcs->bhts", query * scale, key * scale)
        attention = torch.softmax(attention.float(), dim=-1).to(query.dtype)
        attended = torch.einsum("bhts,bhcs->bhct", attention, value)
        attended = attended.reshape(batch, channels, length)
        attended = attended + self.time_projection(time_embedding).to(attended.dtype)[:, :, None]
        return (residual + self.proj_out(attended)).reshape(batch, channels, height, width)


class UNet2D(nn.Module):
    architecture = "film5_bottleneck_mhsa32_v2"

    def __init__(self, base=64):
        super().__init__()
        channels = (base, base * 2, base * 2, base * 4, base * 4)
        time_dim = base * 8
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(base * 2), nn.Linear(base * 2, time_dim),
            nn.SiLU(), nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(1, channels[0], 3, padding=1)
        self.encoder_stages = nn.ModuleList([
            nn.ModuleList([FiLMResBlock(c, c, time_dim), FiLMResBlock(c, c, time_dim)])
            for c in channels
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv2d(c, next_c, 3, stride=2, padding=1)
            for c, next_c in zip(channels, channels[1:])
        ])
        bottleneck = channels[-1]
        self.mid_block1 = FiLMResBlock(bottleneck, bottleneck, time_dim)
        self.mid_attention = TimeConditionedMHSA(bottleneck, time_dim)
        self.mid_block2 = FiLMResBlock(bottleneck, bottleneck, time_dim)
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose2d(c, next_c, 4, stride=2, padding=1)
            for c, next_c in zip(reversed(channels[1:]), reversed(channels[:-1]))
        ])
        self.decoder_stages = nn.ModuleList([
            nn.ModuleList([FiLMResBlock(2 * c, c, time_dim), FiLMResBlock(c, c, time_dim)])
            for c in reversed(channels[:-1])
        ])
        self.out = nn.Sequential(
            _group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], 1, 3, padding=1)
        )

    @staticmethod
    def _run_stage(stage, x, time_embedding):
        for block in stage:
            x = block(x, time_embedding)
        return x

    def forward(self, x, time):
        time_embedding = self.time(time)
        h = self.in_conv(x)
        skips = []
        for stage, downsample in zip(self.encoder_stages[:-1], self.downsamples):
            h = self._run_stage(stage, h, time_embedding)
            skips.append(h)
            h = downsample(h)
        h = self._run_stage(self.encoder_stages[-1], h, time_embedding)
        h = self.mid_block1(h, time_embedding)
        h = self.mid_attention(h, time_embedding)
        h = self.mid_block2(h, time_embedding)
        for upsample, stage, skip in zip(self.upsamples, self.decoder_stages, reversed(skips)):
            h = self._run_stage(stage, torch.cat((upsample(h), skip), dim=1), time_embedding)
        return self.out(h)
