from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        source,
        size=target.shape[2:],
        mode="bilinear",
        align_corners=False,
    )


class REBNCONV(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class RSU7(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.input_conv = REBNCONV(in_ch, out_ch)
        self.enc1 = REBNCONV(out_ch, mid_ch)
        self.enc2 = REBNCONV(mid_ch, mid_ch)
        self.enc3 = REBNCONV(mid_ch, mid_ch)
        self.enc4 = REBNCONV(mid_ch, mid_ch)
        self.enc5 = REBNCONV(mid_ch, mid_ch)
        self.enc6 = REBNCONV(mid_ch, mid_ch)
        self.bottom = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.dec6 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec5 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec4 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs)
        x1 = self.enc1(residual)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.enc5(self.pool(x4))
        x6 = self.enc6(self.pool(x5))
        bottom = self.bottom(x6)
        d6 = self.dec6(torch.cat((bottom, x6), dim=1))
        d5 = self.dec5(torch.cat((_upsample_like(d6, x5), x5), dim=1))
        d4 = self.dec4(torch.cat((_upsample_like(d5, x4), x4), dim=1))
        d3 = self.dec3(torch.cat((_upsample_like(d4, x3), x3), dim=1))
        d2 = self.dec2(torch.cat((_upsample_like(d3, x2), x2), dim=1))
        d1 = self.dec1(torch.cat((_upsample_like(d2, x1), x1), dim=1))
        return d1 + residual


class RSU6(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.input_conv = REBNCONV(in_ch, out_ch)
        self.enc1 = REBNCONV(out_ch, mid_ch)
        self.enc2 = REBNCONV(mid_ch, mid_ch)
        self.enc3 = REBNCONV(mid_ch, mid_ch)
        self.enc4 = REBNCONV(mid_ch, mid_ch)
        self.enc5 = REBNCONV(mid_ch, mid_ch)
        self.bottom = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.dec5 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec4 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs)
        x1 = self.enc1(residual)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.enc5(self.pool(x4))
        bottom = self.bottom(x5)
        d5 = self.dec5(torch.cat((bottom, x5), dim=1))
        d4 = self.dec4(torch.cat((_upsample_like(d5, x4), x4), dim=1))
        d3 = self.dec3(torch.cat((_upsample_like(d4, x3), x3), dim=1))
        d2 = self.dec2(torch.cat((_upsample_like(d3, x2), x2), dim=1))
        d1 = self.dec1(torch.cat((_upsample_like(d2, x1), x1), dim=1))
        return d1 + residual


class RSU5(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.input_conv = REBNCONV(in_ch, out_ch)
        self.enc1 = REBNCONV(out_ch, mid_ch)
        self.enc2 = REBNCONV(mid_ch, mid_ch)
        self.enc3 = REBNCONV(mid_ch, mid_ch)
        self.enc4 = REBNCONV(mid_ch, mid_ch)
        self.bottom = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.dec4 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs)
        x1 = self.enc1(residual)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        bottom = self.bottom(x4)
        d4 = self.dec4(torch.cat((bottom, x4), dim=1))
        d3 = self.dec3(torch.cat((_upsample_like(d4, x3), x3), dim=1))
        d2 = self.dec2(torch.cat((_upsample_like(d3, x2), x2), dim=1))
        d1 = self.dec1(torch.cat((_upsample_like(d2, x1), x1), dim=1))
        return d1 + residual


class RSU4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.input_conv = REBNCONV(in_ch, out_ch)
        self.enc1 = REBNCONV(out_ch, mid_ch)
        self.enc2 = REBNCONV(mid_ch, mid_ch)
        self.enc3 = REBNCONV(mid_ch, mid_ch)
        self.bottom = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs)
        x1 = self.enc1(residual)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bottom = self.bottom(x3)
        d3 = self.dec3(torch.cat((bottom, x3), dim=1))
        d2 = self.dec2(torch.cat((_upsample_like(d3, x2), x2), dim=1))
        d1 = self.dec1(torch.cat((_upsample_like(d2, x1), x1), dim=1))
        return d1 + residual


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.input_conv = REBNCONV(in_ch, out_ch)
        self.enc1 = REBNCONV(out_ch, mid_ch, dilation=1)
        self.enc2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.enc3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.bottom = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch, dilation=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.input_conv(inputs)
        x1 = self.enc1(residual)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        bottom = self.bottom(x3)
        d3 = self.dec3(torch.cat((bottom, x3), dim=1))
        d2 = self.dec2(torch.cat((d3, x2), dim=1))
        d1 = self.dec1(torch.cat((d2, x1), dim=1))
        return d1 + residual


class U2Net(nn.Module):
    """U²-Net penuh dengan deep-supervision side outputs."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.stage1 = RSU7(in_channels, 32, 64)
        self.stage2 = RSU6(64, 32, 128)
        self.stage3 = RSU5(128, 64, 256)
        self.stage4 = RSU4(256, 128, 512)
        self.stage5 = RSU4F(512, 256, 512)
        self.stage6 = RSU4F(512, 256, 512)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x1 = self.stage1(inputs)
        x2 = self.stage2(self.pool12(x1))
        x3 = self.stage3(self.pool23(x2))
        x4 = self.stage4(self.pool34(x3))
        x5 = self.stage5(self.pool45(x4))
        x6 = self.stage6(self.pool56(x5))

        d5 = self.stage5d(torch.cat((_upsample_like(x6, x5), x5), dim=1))
        d4 = self.stage4d(torch.cat((_upsample_like(d5, x4), x4), dim=1))
        d3 = self.stage3d(torch.cat((_upsample_like(d4, x3), x3), dim=1))
        d2 = self.stage2d(torch.cat((_upsample_like(d3, x2), x2), dim=1))
        d1 = self.stage1d(torch.cat((_upsample_like(d2, x1), x1), dim=1))

        side1 = self.side1(d1)
        side2 = _upsample_like(self.side2(d2), side1)
        side3 = _upsample_like(self.side3(d3), side1)
        side4 = _upsample_like(self.side4(d4), side1)
        side5 = _upsample_like(self.side5(d5), side1)
        side6 = _upsample_like(self.side6(x6), side1)
        fused = self.outconv(
            torch.cat((side1, side2, side3, side4, side5, side6), dim=1)
        )
        return fused, side1, side2, side3, side4, side5, side6


def build_resnet50_counter(num_classes: int = 10) -> nn.Module:
    """ResNet50 random-init dengan output kelas jumlah koloni 0–9."""
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}
