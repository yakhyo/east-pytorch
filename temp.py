import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


# def copy_weights(model1, model2):
#     model1.eval()
#     model2.eval()
#     with torch.no_grad():
#         m1_std = model1.state_dict().values()
#         m2_std = model2.state_dict().values()
#         for m1, m2 in zip(m1_std, m2_std):
#             m1.copy_(m2)

# state = {'model': model1.half()}
# torch.save(state, f'weights/best_tf.pt')


# def copy_weights(model1, model2):
#     model1.eval()
#     model2.eval()
#     params1 = model1.parameters()
#     params2 = model2.parameters()
#
#     with torch.no_grad():
#         for param1, param2 in zip(params1, params2):
#             param2.data.copy_(param1.data)


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
        return self.act(self.norm(self.conv(x)))


class VGG16(nn.Module):
    """ VGG16 with batch normalization """

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
        x = self.pool(p5)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    model1 = vgg16_bn(pretrained=True)
    model2 = VGG16()
    copy_weights(model1, model2)

    from PIL import Image

    image = Image.open('cat.jpg')

    from torchvision import transforms

    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)

    model1.eval()
    out = model1(batch_t)
    _, index = torch.max(out, 1)
    print(index)

    model2.eval()
    out = model2(batch_t)
    _, index = torch.max(out, 1)
    print(index)
