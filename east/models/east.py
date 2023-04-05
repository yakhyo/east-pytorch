import math
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(
            self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Conv(nn.Module):
    """Standard Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            inplace: bool = True,
            bias: bool = True,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FeatureExtractor(nn.Module):
    """Feature Extractor stem: VGG16_bn"""

    def __init__(self, cfg: str = "D", weights: Optional[str] = None):
        super().__init__()
        model = VGG(make_layers(cfg=cfgs[cfg], batch_norm=True))
        if weights is not None:
            print("Loading pre-trained weights")
            model.load_state_dict(torch.load(weights))
            print("Done")
        self.features = model.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class FeatureMerge(nn.Module):
    """Feature-merging branch"""

    def __init__(self) -> None:
        super().__init__()
        self.h2 = nn.Sequential(
            Conv(in_channels=1024, out_channels=128, kernel_size=1),
            Conv(in_channels=128, out_channels=128, kernel_size=3),
        )

        self.h3 = nn.Sequential(
            Conv(in_channels=384, out_channels=64, kernel_size=1), Conv(in_channels=64, out_channels=64, kernel_size=3)
        )

        self.h4 = nn.Sequential(
            Conv(in_channels=192, out_channels=32, kernel_size=1), Conv(in_channels=32, out_channels=32, kernel_size=3)
        )

        self.h5 = Conv(in_channels=32, out_channels=32, kernel_size=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: tuple) -> torch.Tensor:
        y = F.interpolate(x[3], scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.h2(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.h3(y)

        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.h4(y)

        y = self.h5(y)

        return y


class Output(nn.Module):
    """Output layer: Detection Head"""

    def __init__(self, scope: int = 512) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1), nn.Sigmoid())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1), nn.Sigmoid())

        self.scope = scope

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        score_map = self.conv1(x)
        location = self.conv2(x) * self.scope
        angle = (self.conv3(x) - 0.5) * math.pi
        geometry = torch.cat([location, angle], 1)

        return score_map, geometry


class EAST(nn.Module):
    """EAST: An Efficient and Accurate Scene Text Detector"""

    def __init__(self, cfg: Optional[str] = "D", weights: Optional[str] = None, scope: int = 512) -> None:
        super().__init__()
        self.extract = FeatureExtractor(cfg=cfg, weights=weights)
        self.merge = FeatureMerge()
        self.detect = Output(scope=scope)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.extract(x)
        x = self.merge(x)
        x = self.detect(x)

        return x
