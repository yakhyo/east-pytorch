import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from util.utils import load_state_dict_from_url

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

model_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvSig(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.conv(x))


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained):
        super(FeatureExtractor, self).__init__()
        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
        if pretrained:
            state_dict = load_state_dict_from_url(model_url)
            vgg16_bn.load_state_dict(state_dict)
        self.features = vgg16_bn.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, nn.MaxPool2d):
                out.append(x)
        return out[1:]


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = Conv(1024, 128, 1)
        self.conv2 = Conv(128, 128, 3, p=1)
        self.conv3 = Conv(384, 64, 1)
        self.conv4 = Conv(64, 64, 3, p=1)
        self.conv5 = Conv(192, 32, 1)
        self.conv6 = Conv(32, 32, 3, p=1)
        self.conv7 = Conv(32, 32, 3, p=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        y = self.conv1(y)
        y = self.conv2(y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.conv3(y)
        y = self.conv4(y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.conv5(y)
        y = self.conv6(y)

        y = self.conv7(y)
        return y


class Head(nn.Module):
    def __init__(self, scope=512):
        super(Head, self).__init__()
        self.conv1 = ConvSig(32, 1, 1)
        self.conv2 = ConvSig(32, 4, 1)
        self.conv3 = ConvSig(32, 1, 1)

        self.scope = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.conv1(x)
        loc =   self.conv2(x) * self.scope
        angle = (self.conv3(x) - 0.5) * math.pi
        geo =   torch.cat((loc, angle), 1)
        return  score, geo


class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = FeatureExtractor(pretrained)
        self.merge = Backbone()
        self.output = Head()

    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))


if __name__ == '__main__':
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)
