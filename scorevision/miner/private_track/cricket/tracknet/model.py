"""
TrackNetV2 model (PyTorch) — VGG16-style encoder/decoder.

Input : N consecutive frames stacked on the channel axis (default 3 frames RGB
        => 9 channels), shape (B, 3*F, H, W).
Output: F heatmaps (one per input frame), shape (B, F, H, W), sigmoid in [0,1].

Determinism note: the spotcheck requires reproducible inference (threshold 0.98).
At inference set torch deterministic flags + FP32 + fixed input pipeline. This
module has no internal randomness (dropout-free); BatchNorm runs in eval mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _cbr(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_c),
    )


class TrackNetV2(nn.Module):
    def __init__(self, n_frames: int = 3, in_ch_per_frame: int = 3, out_ch: int | None = None):
        super().__init__()
        c = n_frames * in_ch_per_frame
        self.n_frames = n_frames
        self.out_ch = out_ch or n_frames

        # encoder
        self.e1 = nn.Sequential(_cbr(c, 64), _cbr(64, 64))
        self.e2 = nn.Sequential(_cbr(64, 128), _cbr(128, 128))
        self.e3 = nn.Sequential(_cbr(128, 256), _cbr(256, 256), _cbr(256, 256))
        self.e4 = nn.Sequential(_cbr(256, 512), _cbr(512, 512), _cbr(512, 512))
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # decoder (concat skip connections)
        self.d3 = nn.Sequential(_cbr(512 + 256, 256), _cbr(256, 256), _cbr(256, 256))
        self.d2 = nn.Sequential(_cbr(256 + 128, 128), _cbr(128, 128))
        self.d1 = nn.Sequential(_cbr(128 + 64, 64), _cbr(64, 64))
        self.head = nn.Conv2d(64, self.out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)                 # H
        x2 = self.e2(self.pool(x1))     # H/2
        x3 = self.e3(self.pool(x2))     # H/4
        x4 = self.e4(self.pool(x3))     # H/8

        u3 = self.up(x4)                # H/4
        u3 = self.d3(torch.cat([u3, x3], dim=1))
        u2 = self.up(u3)                # H/2
        u2 = self.d2(torch.cat([u2, x2], dim=1))
        u1 = self.up(u2)                # H
        u1 = self.d1(torch.cat([u1, x1], dim=1))
        return torch.sigmoid(self.head(u1))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TrackNetV2(n_frames=3).eval()
    x = torch.randn(1, 9, 288, 512)
    with torch.no_grad():
        y = m(x)
    print("output", tuple(y.shape), "params(M)", round(count_params(m) / 1e6, 2))
    assert y.shape == (1, 3, 288, 512)
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    print("model self-check passed")
