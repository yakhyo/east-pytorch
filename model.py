import math
import torch
import torch.nn as nn
import torch.nn.functional


# def copy_weights(model1, model2):
#     model1.eval()
#     model2.eval()
#     with torch.no_grad():
#         m1_std = model1.state_dict().values()
#         m2_std = model2.state_dict().values()
#         for m1, m2 in zip(m1_std, m2_std):
#             m1.copy_(m2)

def _init_weights(self):
    """ Standard weight initializer """
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


class Interpolate(nn.Module):
    """ Wrapper class for `nn.functional.interpolate()` """

    def __init__(self, scale_factor: int, mode: str = 'bilinear', align_corners: bool = None) -> None:
        super().__init__()
        self.fn = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fn(input=x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x


class Conv(nn.Module):
    """ Standard convolution module """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm: bool = False,
            act: str = None,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.norm = nn.BatchNorm2d(num_features=out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class VGG16(nn.Module):
    """ Feature extractor stem: VGG16 with batch normalization """

    def __init__(self, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        super().__init__()

        # p1/2
        self.p1 = nn.Sequential(
            Conv(in_channels=3, out_channels=64, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, norm=True, act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # p2/4
        self.p2 = nn.Sequential(
            Conv(in_channels=64, out_channels=128, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=128, out_channels=128, kernel_size=3, padding=1, norm=True, act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # p3/8
        self.p3 = nn.Sequential(
            Conv(in_channels=128, out_channels=256, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=256, out_channels=256, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=256, out_channels=256, kernel_size=3, padding=1, norm=True, act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # p4/16
        self.p4 = nn.Sequential(
            Conv(in_channels=256, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=512, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=512, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # p5/32
        self.p5 = nn.Sequential(
            Conv(in_channels=512, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=512, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            Conv(in_channels=512, out_channels=512, kernel_size=3, padding=1, norm=True, act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

        if init_weights:
            _init_weights(self)

    def forward(self, x: torch.Tensor) -> tuple:
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return p2, p3, p4, p5


class FeatureMerge(nn.Module):
    """ Feature-merging branch """

    def __init__(self, init_weights: bool = True) -> None:
        super().__init__()
        self.interpolate = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        # stage 1
        self.stage1 = nn.Sequential(
            Conv(in_channels=1024, out_channels=128, kernel_size=1, norm=True, act='relu'),
            Conv(in_channels=128, out_channels=128, kernel_size=3, padding=1, norm=True, act='relu'),
        )
        # stage 2
        self.stage2 = nn.Sequential(
            Conv(in_channels=384, out_channels=64, kernel_size=1, norm=True, act='relu'),
            Conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, norm=True, act='relu'),
        )
        # stage 3
        self.stage3 = nn.Sequential(
            Conv(in_channels=192, out_channels=32, kernel_size=1, norm=True, act='relu'),
            Conv(in_channels=32, out_channels=32, kernel_size=3, padding=1, norm=True, act='relu'),
        )
        # stage 4
        self.stage4 = Conv(in_channels=32, out_channels=32, kernel_size=3, padding=1, norm=True, act='relu')
        if init_weights:
            _init_weights(self)

    def forward(self, x: tuple) -> torch.Tensor:
        p2, p3, p4, p5 = x

        x = self.interpolate(p5)
        x = torch.cat([x, p4], 1)
        x = self.stage1(x)

        x = self.interpolate(x)
        x = torch.cat([x, p3], 1)
        x = self.stage2(x)

        x = self.interpolate(x)
        x = torch.cat([x, p2], 1)
        x = self.stage3(x)

        x = self.stage4(x)

        return x


class Head(nn.Module):
    """ Output layer: Detection Head """

    def __init__(self, scope: int = 512, init_weight: bool = True) -> None:
        super().__init__()

        self.conv1 = Conv(in_channels=32, out_channels=1, kernel_size=1, padding=0, act='sigmoid')
        self.conv2 = Conv(in_channels=32, out_channels=4, kernel_size=1, padding=0, act='sigmoid')
        self.conv3 = Conv(in_channels=32, out_channels=1, kernel_size=1, padding=0, act='sigmoid')
        self.scope = scope

        if init_weight:
            _init_weights(self)

    def forward(self, x: torch.Tensor) -> tuple:
        score = self.conv1(x)
        loc = self.conv2(x) * self.scope
        angle = (self.conv3(x) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)

        return score, geo


class EAST(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = VGG16()
        self.merge = FeatureMerge()
        self.detect = Head()

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.backbone(x)
        x = self.merge(x)
        x = self.detect(x)

        return x


if __name__ == '__main__':
    net = EAST()
    dummy = torch.randn(1, 3, 256, 256)
    score, geo = net(dummy)
    print(score.shape)
    print(geo.shape)
