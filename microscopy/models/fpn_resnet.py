import torch.nn as nn


__all__ = ['FPNResNet', 'fpn_resnet50', 'fpn_resnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FPNResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, lateral_channels=[2048, 1024, 512, 256], mid_channels=256):
        super(FPNResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.uprefine5 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.c5_lateral = nn.Conv2d(
            lateral_channels[0],
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.c4_lateral = nn.Conv2d(
            lateral_channels[1],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c3_lateral = nn.Conv2d(
            lateral_channels[2],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c2_lateral = nn.Conv2d(
            lateral_channels[3],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.conv_after_sum4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s4 = norm_layer(mid_channels)
        self.conv_after_sum3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s3 = norm_layer(mid_channels)
        self.conv_after_sum2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s2 = norm_layer(mid_channels)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.uprefine5 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.uprefine3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.c5_lateral = nn.Conv2d(
            lateral_channels[0],
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.c4_lateral = nn.Conv2d(
            lateral_channels[1],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c3_lateral = nn.Conv2d(
            lateral_channels[2],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.c2_lateral = nn.Conv2d(
            lateral_channels[3],
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.conv_after_sum4 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s4 = norm_layer(mid_channels)
        self.conv_after_sum3 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s3 = norm_layer(mid_channels)
        self.conv_after_sum2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bn_s2 = norm_layer(mid_channels)

        #self.predict5 = nn.Conv2d(
        #    mid_channels,
        #    num_classes+1,
        #    kernel_size=3,
        #    stride=1,
        #    padding=1)
        #self.predict4 = nn.Conv2d(
        #    mid_channels,
        #    num_classes+1,
        #    kernel_size=3,
        #    stride=1,
        #    padding=1)
        #self.predict3 = nn.Conv2d(
        #    mid_channels,
        #    num_classes+1,
        #    kernel_size=3,
        #    stride=1,
        #    padding=1)
        self.predict2 = nn.Conv2d(
            mid_channels,
            num_classes+1,
            kernel_size=3,
            stride=1,
            padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.relu(self.c5_lateral(c5))

        p5_up = self.uprefine5(self.upsample5(p5))
        c4_lat = self.c4_lateral(c4)
        p4 = self.relu(self.bn_s4(self.conv_after_sum4(c4_lat + p5_up)))

        p4_up = self.uprefine4(self.upsample4(p4))
        c3_lat = self.c3_lateral(c3)
        p3 = self.relu(self.bn_s3(self.conv_after_sum3(c3_lat + p4_up)))

        p3_up = self.uprefine3(self.upsample3(p3))
        c2_lat = self.c2_lateral(c2)
        p2 = self.relu(self.bn_s2(self.conv_after_sum2(c2_lat + p3_up)))

        #pred5 = self.predict5(p5)
        #pred4 = self.predict4(p4)
        #pred3 = self.predict3(p3)
        pred2 = self.predict2(p2)

        gx = self.avgpool(c5)
        gx = gx.reshape(x.size(0), -1)
        gx = self.fc(gx)

        return gx, pred2


def resnet50_fpn(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return FPNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_fpn(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return FPNResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
