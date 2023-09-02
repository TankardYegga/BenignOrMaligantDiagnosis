import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
# from torch.utils.model_zoo import model_zoo
from test_bottleneck import ACmix
from torchvision import models

device_str = "cuda:0"

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)  # 1* 1卷积中stride设置不等于1即可
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, k_att, head, k_conv, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 对应3*3的卷积，且包含下采样步骤
        self.conv2 = ACmix(width, width, k_att, head, k_conv, stride=stride, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)

        # print(self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, k_att=7, head=4, k_conv=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        # 设置默认的norm方式
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

        self.layer1 = self._make_layer(block, 64, layers[0], k_att, head, k_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], k_att, head, k_conv, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], k_att, head, k_conv, stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], k_att, head, k_conv, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, rate, k, head, stride=1, dilate=False):
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
        # 只有第一个block里面进行了通道数的扩充，在其他block里面则保持通道数目不变
        # 只有第一个block里面需要传入stride参数，在其他block里面则不需要传入stride参数，也可以说是默认stride=1
        # 只有第一个block里面传入了downsample参数，在其他block里面则不需要传入，也就是说默认只对第一个block进行下采样
        # 这里的rate、k、和head分别对应BottleNeck中Acmix模块的k_att, head, k_conv的三个参数
        layers.append(block(self.inplanes, planes, rate, k, head, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate, k, head, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetSmall(nn.Module):

    def __init__(self,  block, extra_model, layers, k_att=7, head=4, k_conv=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSmall, self).__init__()

        self.extra_model = nn.Sequential(*[extra_model.layer3, extra_model.layer4, extra_model.avgpool]);
        self.extra_model_fc = extra_model.fc;

        # 设置默认的norm方式
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

        self.layer1 = self._make_layer(block, 64, layers[0], k_att, head, k_conv)
        self.layer2 = self._make_layer(block, 128, layers[1], k_att, head, k_conv, stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, rate, k, head, stride=1, dilate=False):
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
        # 只有第一个block里面进行了通道数的扩充，在其他block里面则保持通道数目不变
        # 只有第一个block里面需要传入stride参数，在其他block里面则不需要传入stride参数，也可以说是默认stride=1
        # 只有第一个block里面传入了downsample参数，在其他block里面则不需要传入，也就是说默认只对第一个block进行下采样
        # 这里的rate、k、和head分别对应BottleNeck中Acmix模块的k_att, head, k_conv的三个参数
        layers.append(block(self.inplanes, planes, rate, k, head, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate, k, head, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) # （2,512,32,32）

        x = self.extra_model(x)
        x = torch.flatten(x, 1)
        x = self.extra_model_fc(x)

        # x = self.avgpool(x)  # （2,512,1,1）
        # x = torch.flatten(x, 1)  # （2,512）
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetSmallSmall(nn.Module):

    def __init__(self,  block, extra_model, layers, k_att=7, head=4, k_conv=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetSmallSmall, self).__init__()

        self.extra_model = nn.Sequential(*[extra_model.layer2, extra_model.layer3, extra_model.layer4, extra_model.avgpool])
        self.extra_model_fc = extra_model.fc

        # 设置默认的norm方式
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

        self.layer1 = self._make_layer(block, 64, layers[0], k_att, head, k_conv)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, rate, k, head, stride=1, dilate=False):
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
        # 只有第一个block里面进行了通道数的扩充，在其他block里面则保持通道数目不变
        # 只有第一个block里面需要传入stride参数，在其他block里面则不需要传入stride参数，也可以说是默认stride=1
        # 只有第一个block里面传入了downsample参数，在其他block里面则不需要传入，也就是说默认只对第一个block进行下采样
        # 这里的rate、k、和head分别对应BottleNeck中Acmix模块的k_att, head, k_conv的三个参数
        layers.append(block(self.inplanes, planes, rate, k, head, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate, k, head, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.extra_model(x)
        x = torch.flatten(x, 1)
        x = self.extra_model_fc(x)

        # x = self.avgpool(x)  # 
        # x = torch.flatten(x, 1)  # 
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def ACmix_ResNet(layers=[3,4,6,3], **kwargs):
    return _resnet(Bottleneck, layers, **kwargs)


def _resnet_small(block, extra_model, layers, **kwargs):
    model = ResNetSmall(block, extra_model, layers, **kwargs)
    return model


def ACmix_ResNet_Small(extra_model, layers=[3,4,6,3], **kwargs):
    return _resnet_small(Bottleneck, extra_model, layers, **kwargs)


def _resnet_small_small(block, extra_model, layers, **kwargs):
    model = ResNetSmallSmall(block, extra_model, layers, **kwargs)
    return model


def ACmix_ResNet_Small_Small(extra_model, layers=[3,4,6,3], **kwargs):
    return _resnet_small_small(Bottleneck, extra_model, layers, **kwargs)


# if __name__ == '__main__':
    # resnet50
    # model = ACmix_ResNet(layers=[3,4,6,3], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # # resnet18
    # model = ACmix_ResNet(layers=[2,2,2,2], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # # resnet26
    # model = ACmix_ResNet(layers=[2,2,2,2], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # # resnet38
    # model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # # resnet101
    # model = ACmix_ResNet(layers=[3,4,23,3], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)
    
    # # resnet152
    # model = ACmix_ResNet(layers=[3,8,36,3], num_classes=2).cuda()
    # # input = torch.randn([2,3,224,224]).cuda()
    # input = torch.randn([2,3,256,256]).cuda()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)


    # resnet38  减小head的值
    # model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2, k_att=7, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3,256,256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # resnet38  减小head和k_att的值
    # model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2, k_att=5, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3,256,256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # resnet38  减小head和k_att的值
    # model = ACmix_ResNet(layers=[3,3,3,3], num_classes=2, k_att=3, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3,256,256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # resnet10
    # model = ACmix_ResNet(layers=[1,1,1,1], num_classes=2, k_att=7, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3,256,256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # extra_model = models.resnet50(pretrained=True)
    # extra_model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    # model = ACmix_ResNet_Small(extra_model, layers=[3, 3], num_classes=2, k_att=7, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3, 256, 256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)

    # extra_model = models.resnet50(pretrained=True)
    # extra_model.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    # model = ACmix_ResNet_Small_Small(extra_model, layers=[3, 3], num_classes=2, k_att=7, head=1, k_conv=3).cuda(device_str)
    # input = torch.randn([2,3, 256, 256]).cuda(device_str)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(model(input).shape)
    
