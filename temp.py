import math
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def _init_weights(self):
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


class Conv2dNormActivation(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 1,
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
        return self.act(self.norm(self.conv(x)))


class VGG16(nn.Module):

    def __init__(self, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5):
        super().__init__()

        # p1/2
        self.p1 = nn.Sequential(
            Conv2dNormActivation(in_channels=3, out_channels=64, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=64, out_channels=64, kernel_size=3, norm=False, act='relu', bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # p2/4
        self.p2 = nn.Sequential(
            Conv2dNormActivation(in_channels=64, out_channels=128, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=128, out_channels=128, kernel_size=3, norm=False, act='relu', bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # p3/8
        self.p3 = nn.Sequential(
            Conv2dNormActivation(in_channels=128, out_channels=256, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=256, out_channels=256, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=256, out_channels=256, kernel_size=3, norm=False, act='relu', bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # p4/16
        self.p4 = nn.Sequential(
            Conv2dNormActivation(in_channels=256, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=512, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=512, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # p5/32
        self.p5 = nn.Sequential(
            Conv2dNormActivation(in_channels=512, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=512, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
            Conv2dNormActivation(in_channels=512, out_channels=512, kernel_size=3, norm=False, act='relu', bias=True),
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
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        p2, p3, p4, p5 = x
        y = F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, p4), 1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, p3), 1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, p2), 1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))
        return y


class Head(nn.Module):
    def __init__(self, scope: int = 512, init_weight: bool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()

        self.scope = scope

        if init_weight:
            _init_weights(self)

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        return score, geo


class EAST(nn.Module):
    def __init__(self, pretrained=False):
        super(EAST, self).__init__()
        self.backbone = VGG16()
        self.merge = FeatureMerge()
        self.detect = Head()

    def forward(self, x):
        return self.detect(self.merge(self.backbone(x)))


if __name__ == '__main__':
    # ggg = VGG16()
    # print(ggg)
    # print(sum(p.numel() for p in ggg.parameters() if p.requires_grad))
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)
