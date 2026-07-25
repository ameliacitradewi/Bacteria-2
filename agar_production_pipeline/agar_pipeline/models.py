from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class REBNCONV(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RSU7(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(self.pool5(hx5))
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), dim=1))
        hx5d = self.rebnconv5d(torch.cat((_upsample_like(hx6d, hx5), hx5), dim=1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), dim=1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(self.pool4(hx4))
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), dim=1))
        hx4d = self.rebnconv4d(torch.cat((_upsample_like(hx5d, hx4), hx4), dim=1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(self.pool3(hx3))
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), dim=1))
        hx3d = self.rebnconv3d(torch.cat((_upsample_like(hx4d, hx3), hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(self.pool1(hx1))
        hx3 = self.rebnconv3(self.pool2(hx2))
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilation=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), dim=1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), dim=1))
        return hx1d + hxin


class U2NETP(nn.Module):
    """Small U²-Net with deep-supervision outputs.

    Training returns ``(d0, d1, ..., d6)`` logits. Production wrappers should
    keep only ``d0`` and apply sigmoid outside the base model.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1) -> None:
        super().__init__()
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(64, 16, 64)

        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(out_ch * 6, out_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        hx5d = self.stage5d(torch.cat((_upsample_like(hx6, hx5), hx5), dim=1))
        hx4d = self.stage4d(torch.cat((_upsample_like(hx5d, hx4), hx4), dim=1))
        hx3d = self.stage3d(torch.cat((_upsample_like(hx4d, hx3), hx3), dim=1))
        hx2d = self.stage2d(torch.cat((_upsample_like(hx3d, hx2), hx2), dim=1))
        hx1d = self.stage1d(torch.cat((_upsample_like(hx2d, hx1), hx1), dim=1))

        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), dim=1))
        return d0, d1, d2, d3, d4, d5, d6


class U2NetProductionWrapper(nn.Module):
    def __init__(self, model: U2NETP) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x)[0])


class ConvHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNet50FPNCenterNet(nn.Module):
    """Single-class CenterNet-style detector with a ResNet50-FPN backbone.

    The neural network emits three dense tensors at stride 4:
    ``heatmap_logits``, ``size_raw``, and ``offset_raw``. Decoding and NMS are
    intentionally outside the model so the exported Core ML graph stays simple.
    """

    output_stride: int = 4

    def __init__(
        self,
        fpn_channels: int = 128,
        pretrained: bool = True,
        freeze_stem: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            print(
                "[WARNING] Bobot ImageNet ResNet50 tidak dapat dimuat; "
                f"melanjutkan dari random initialization: {exc}"
            )
            backbone = resnet50(weights=None)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lateral2 = nn.Conv2d(256, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(512, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(1024, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(2048, fpn_channels, 1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = ConvHead(fpn_channels, 1, hidden=fpn_channels)
        self.size_head = ConvHead(fpn_channels, 2, hidden=fpn_channels)
        self.offset_head = ConvHead(fpn_channels, 2, hidden=fpn_channels)
        nn.init.constant_(self.heatmap_head.block[-1].bias, -2.19)

        if freeze_stem:
            for parameter in self.stem.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + _upsample_like(p5, c4)
        p3 = self.lateral3(c3) + _upsample_like(p4, c3)
        p2 = self.lateral2(c2) + _upsample_like(p3, c2)

        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        fused = self.fuse(
            p2
            + _upsample_like(p3, p2)
            + _upsample_like(p4, p2)
            + _upsample_like(p5, p2)
        )
        return {
            "heatmap_logits": self.heatmap_head(fused),
            "size_raw": self.size_head(fused),
            "offset_raw": self.offset_head(fused),
        }


class ColonyProductionWrapper(nn.Module):
    """Core ML-friendly wrapper with constrained output ranges."""

    def __init__(self, model: ResNet50FPNCenterNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(x)
        heatmap = torch.sigmoid(outputs["heatmap_logits"])
        size = F.softplus(outputs["size_raw"])
        offset = torch.sigmoid(outputs["offset_raw"])
        return heatmap, size, offset


def load_state_dict_flexible(
    model: nn.Module,
    checkpoint: dict[str, object] | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    if isinstance(checkpoint, str):
        payload = torch.load(checkpoint, map_location=map_location)
    else:
        payload = checkpoint
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint harus berupa dictionary.")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("model_state_dict tidak ditemukan.")
    clean_state = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(clean_state, strict=True)
    return payload


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def set_backbone_trainable(
    model: ResNet50FPNCenterNet,
    trainable_layers: Sequence[str],
) -> None:
    trainable = set(trainable_layers)
    for name in ("stem", "layer1", "layer2", "layer3", "layer4"):
        module = getattr(model, name)
        requires_grad = name in trainable
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad
