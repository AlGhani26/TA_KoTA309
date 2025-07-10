import torch
import torch.nn as nn
import torch.nn.functional as F

class SKConv(nn.Module):
    def __init__(self, channels, stride=1, groups=32, reduction=16):
        super(SKConv, self).__init__()
        d = max(channels // reduction, 4)

        # Two separate convolutional branches
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=2, dilation=2, groups=groups, bias=False)
        self.bn5 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        # Squeeze-and-Excitation (SK attention)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, d, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(d, channels * 2, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x3 = self.relu(self.bn3(self.conv3(x)))  # branch 1
        x5 = self.relu(self.bn5(self.conv5(x)))  # branch 2

        feats = torch.stack([x3, x5], dim=1)  # shape: (B, 2, C, H, W)

        # Fuse
        u = x3 + x5
        s = self.global_pool(u)
        z = self.fc1(s)
        a_b = self.fc2(z).view(x.size(0), 2, x.size(1), 1, 1)  # shape: (B, 2, C, 1, 1)
        a_b = self.softmax(a_b)

        out = (feats * a_b).sum(dim=1)
        return out


class resnext_block(nn.Module):
    def __init__(self, in_channels, cardinality, bwidth, idt_downsample=None, stride=1):
        super(resnext_block, self).__init__()
        self.expansion = 2
        out_channels = cardinality * bwidth

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Replaced conv2 with SKConv
        self.skconv = SKConv(out_channels, stride=stride, groups=cardinality)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = idt_downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.skconv(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)
        return out

# == SKResNeXt50 Encoder ==
class SKResNeXt(nn.Module):
    def __init__(self, resnet_block, layers, cardinality, bwidth, img_channels, num_classes):
        super(SKResNeXt, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.bwidth = bwidth

        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._layers(resnet_block, layers[0], stride=1)
        self.layer2 = self._layers(resnet_block, layers[1], stride=2)
        self.layer3 = self._layers(resnet_block, layers[2], stride=2)
        self.layer4 = self._layers(resnet_block, layers[3], stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x0p = self.maxpool(x0)
        x1 = self.layer1(x0p)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4

    def _layers(self, block, blocks, stride):
        identity_downsample = None
        out_channels = self.cardinality * self.bwidth
        layers = []

        if stride != 1 or self.in_channels != out_channels * 2:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 2)
            )

        layers.append(block(self.in_channels, self.cardinality, self.bwidth, identity_downsample, stride))
        self.in_channels = out_channels * 2

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, self.cardinality, self.bwidth))

        self.bwidth *= 2
        return nn.Sequential(*layers)

def SKResNeXt50(img_channels=4, num_classes=6, cardinality=32, bwidth=4):
    return SKResNeXt(resnext_block, [3, 4, 6, 3], cardinality, bwidth, img_channels, num_classes)