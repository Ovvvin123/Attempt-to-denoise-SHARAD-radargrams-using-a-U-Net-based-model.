# model_unet.py
#   输入 noisy patch:
#       x.shape = [B, 1, H, W]
#
#   网络输出 predicted_noise:
#       predicted_noise.shape = [B, 1, H, W]
#


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_count(num_channels: int, max_groups: int = 8) -> int:
    """
    为 GroupNorm 自动选择 group 数。
    要求：
        num_channels % num_groups == 0

    例如：
        channels=32 -> groups=8
        channels=64 -> groups=8
        channels=16 -> groups=8
        channels=8  -> groups=8
        channels=4  -> groups=4
    """
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def make_norm(norm: str, num_channels: int) -> nn.Module:
    """
    创建归一化层。

    """
    norm = norm.lower()

    if norm == "group":
        num_groups = get_group_count(num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    if norm == "batch":
        return nn.BatchNorm2d(num_channels)

    if norm == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)

    if norm == "none":
        return nn.Identity()

    raise ValueError(f"未知 norm 类型：{norm}")


class ConvBlock(nn.Module):
    """
    基础卷积模块：

        Conv2d -> Norm -> ReLU
        Conv2d -> Norm -> ReLU

    输入：
        [B, C_in, H, W]

    输出：
        [B, C_out, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(norm, out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                make_norm(norm, out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    下采样模块：

        MaxPool2d(2)
        ConvBlock

    输入：
        [B, C_in, H, W]

    输出：
        [B, C_out, H/2, W/2]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm=norm,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    上采样模块：

        bilinear upsample
        concat skip connection
        ConvBlock

    输入：
        x    : 来自更深层的特征
        skip : encoder 中对应尺度的特征

    输出：
        融合 skip 之后的特征
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str = "group",
        dropout: float = 0.0,
        up_mode: str = "bilinear",
    ):
        super().__init__()

        up_mode = up_mode.lower()

        if up_mode == "bilinear":
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
            conv_in_channels = out_channels + skip_channels

        elif up_mode == "transpose":
            self.up = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
            )
            conv_in_channels = out_channels + skip_channels

        else:
            raise ValueError(f"未知 up_mode：{up_mode}")

        self.conv = ConvBlock(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            norm=norm,
            dropout=dropout,
        )

    @staticmethod
    def match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        让 x 的 H, W 和 target 一致。

        """
        target_h, target_w = target.shape[-2:]
        x_h, x_w = x.shape[-2:]

        diff_h = target_h - x_h
        diff_w = target_w - x_w

        if diff_h == 0 and diff_w == 0:
            return x

        # 如果 x 比 target 小，就 padding
        pad_top = max(diff_h // 2, 0)
        pad_bottom = max(diff_h - pad_top, 0)
        pad_left = max(diff_w // 2, 0)
        pad_right = max(diff_w - pad_left, 0)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

        # 如果 x 比 target 大，就中心裁剪
        x_h, x_w = x.shape[-2:]

        if x_h > target_h:
            start = (x_h - target_h) // 2
            x = x[:, :, start:start + target_h, :]

        if x_w > target_w:
            start = (x_w - target_w) // 2
            x = x[:, :, :, start:start + target_w]

        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.match_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "group",
        up_mode: str = "bilinear",
        dropout: float = 0.0,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth 必须 >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.norm = norm
        self.up_mode = up_mode
        self.dropout = dropout

        # encoder channels:
        # depth=3, base=32 -> [32, 64, 128, 256]
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.input_block = ConvBlock(
            in_channels=in_channels,
            out_channels=channels[0],
            norm=norm,
            dropout=dropout,
        )

        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.down_blocks.append(
                DownBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    norm=norm,
                    dropout=dropout,
                )
            )

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_blocks.append(
                UpBlock(
                    in_channels=channels[i + 1],
                    skip_channels=channels[i],
                    out_channels=channels[i],
                    norm=norm,
                    dropout=dropout,
                    up_mode=up_mode,
                )
            )

        self.output_conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
        )

        self._init_weights()

    def _init_weights(self):
        """
        初始化卷积层参数。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x: [B, 1, H, W]

        输出：
            predicted_noise: [B, 1, H, W]
        """
        skips = []

        x = self.input_block(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = skips[-1]
        skips = skips[:-1]

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        predicted_noise = self.output_conv(x)

        return predicted_noise


def count_parameters(model: nn.Module) -> int:
    """
    统计可训练参数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """
    简单测试模型输入输出尺寸。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=3,
        norm="group",
        up_mode="bilinear",
        dropout=0.0,
    ).to(device)

    print("Device:", device)
    print("Model:", model.__class__.__name__)
    print("Trainable parameters:", count_parameters(model))

    test_shapes = [
        (2, 1, 176, 256),
        (2, 1, 176, 512),
        (1, 1, 176, 200),
    ]

    model.eval()

    with torch.no_grad():
        for shape in test_shapes:
            x = torch.randn(*shape, device=device)
            predicted_noise = model(x)
            denoised = x - predicted_noise

            print("\nInput shape:          ", tuple(x.shape))
            print("Predicted noise shape:", tuple(predicted_noise.shape))
            print("Denoised shape:       ", tuple(denoised.shape))

            assert predicted_noise.shape == x.shape, "输出尺寸必须和输入尺寸一致"
            assert denoised.shape == x.shape, "denoised 尺寸必须和输入尺寸一致"

    print("\nmodel_unet.py 测试通过。")


if __name__ == "__main__":
    test_model()