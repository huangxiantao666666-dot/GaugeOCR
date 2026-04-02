"""
DeepLabV3+ Complete Implementation
==================================
A complete implementation of the DeepLabV3+ architecture for semantic image segmentation.
This implementation includes both ResNet and MobileNetV2 backbones, as well as methods
for loading pretrained weights.

DeepLabV3+ is an encoder-decoder architecture that uses atrous convolution to capture
multi-scale contextual information and a decoder module to recover spatial details.

References:
- [1] Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017).
      Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587.
- [2] Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).
      Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.
      In Proceedings of the European Conference on Computer Vision (ECCV).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os
from PIL import Image


# ============================================
# Utility Functions and Classes
# ============================================

class _SimpleSegmentationModel(nn.Module):
    """
    A simple base class for segmentation models that combines a backbone network
    with a classifier head, and handles the final interpolation to match input size.
    
    Args:
        backbone: The feature extraction backbone network
        classifier: The segmentation classifier/decoder network
    """
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        """
        Forward pass through the segmentation model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output segmentation map of shape [batch_size, num_classes, height, width],
            interpolated to match the input spatial dimensions.
            
        Tensor Shape Changes:
            x: [B, 3, H, W] -> backbone(features) -> [dict with multiple feature maps]
            features: dict -> classifier -> [B, num_classes, H/4, W/4] (typically 1/4 of input size)
            interpolated: [B, num_classes, H/4, W/4] -> [B, num_classes, H, W]
        """
        input_shape = x.shape[-2:]  # Save input spatial dimensions: (H, W)
        features = self.backbone(x)  # Extract features: returns dict with 'out' and 'low_level' keys
        x = self.classifier(features)  # Apply segmentation head: [B, num_classes, H/4, W/4]
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)  # Upsample to original size: [B, num_classes, H, W]
        return x


class IntermediateLayerGetter(nn.ModuleDict): # nn.ModuleDict can be used to store multiple modules
    """
    Module wrapper that returns intermediate layers from a model.
    
    This class allows extracting features from specific layers of a pre-trained
    model, which is essential for encoder-decoder architectures that need both
    high-level semantic features and low-level spatial features.
    
    Args:
        model: The model from which to extract intermediate layers
        return_layers: A dictionary mapping layer names in the model to output names
                      e.g. {'layer4': 'out', 'layer1': 'low_level'}
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        # Initialize the ModuleDict with the selected layers
        # This will make self.named_children() return the selected layers
        super(IntermediateLayerGetter, self).__init__(layers) 
        self.return_layers = orig_return_layers

    def forward(self, x):
        """
        Forward pass that returns the specified intermediate layers.
        
        Args:
            x: Input tensor
            
        Returns:
            OrderedDict containing the outputs from the specified layers
        """
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


# ============================================
# Backbone: ResNet
# ============================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding and dilation.
    
    Creates a 3x3 convolutional layer that maintains spatial dimensions
    when stride=1 and dilation is properly set.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Convolution stride
        groups: Number of groups for grouped convolution
        dilation: Dilation rate for atrous convolution
        
    Returns:
        Conv2d layer with the specified parameters
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution.
    
    Creates a 1x1 convolutional layer typically used for channel dimension
    reduction or projection.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        stride: Convolution stride
        
    Returns:
        Conv2d layer with the specified parameters
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet architectures.
    
    The bottleneck design reduces computational complexity by using 1x1 convolutions
    to reduce and then restore channel dimensions, with a 3x3 convolution in between.
    
    Args:
        inplanes: Number of input channels
        planes: Number of intermediate channels (output channels will be planes * expansion)
        stride: Convolution stride for the middle 3x3 conv
        downsample: Optional downsampling module for the residual connection
        groups: Number of groups for grouped convolution
        base_width: Base width for width scaling
        dilation: Dilation rate for atrous convolution
        norm_layer: Normalization layer to use (default: BatchNorm2d)
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
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
        """
        Forward pass through the bottleneck block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying the bottleneck transformation
            
        Tensor Shape Changes:
            x: [B, C_in, H, W]
            conv1: [B, C_in, H, W] -> [B, width, H, W] (1x1 conv, channel reduction)
            conv2: [B, width, H, W] -> [B, width, H', W'] (3x3 conv, may downsample if stride>1)
            conv3: [B, width, H', W'] -> [B, C_out, H', W'] (1x1 conv, channel expansion)
            where C_out = planes * expansion (typically 4x of planes)
        """
        identity = x  # Save input for residual connection: [B, C_in, H, W]

        out = self.conv1(x)  # 1x1 conv: [B, C_in, H, W] -> [B, width, H, W]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # 3x3 conv: [B, width, H, W] -> [B, width, H', W']
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 1x1 conv: [B, width, H', W'] -> [B, C_out, H', W']
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # Match dimensions for residual: [B, C_in, H, W] -> [B, C_out, H', W']

        out += identity  # Residual connection: element-wise addition
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet backbone network.
    
    Implements the ResNet architecture with support for dilated convolutions,
    which is essential for dense prediction tasks like semantic segmentation.
    
    Args:
        block: The residual block type to use (e.g., Bottleneck)
        layers: Number of blocks in each of the 4 layers
        num_classes: Number of output classes for classification (not used in segmentation)
        zero_init_residual: Whether to zero-initialize the last BN in each residual branch
        groups: Number of groups for grouped convolutions
        width_per_group: Base width per group
        replace_stride_with_dilation: Whether to replace strides with dilation in layers 2-4
        norm_layer: Normalization layer to use
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Creates a layer consisting of multiple residual blocks.
        
        Args:
            block: The residual block type
            planes: Number of channels for the blocks
            blocks: Number of blocks in this layer
            stride: Stride for the first block
            dilate: Whether to use dilation instead of stride
            
        Returns:
            Sequential module containing the layer blocks
        """
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
        """
        Forward pass through the ResNet.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Output features from the final layer
            
        Tensor Shape Changes (assuming input [B, 3, H, W]):
            conv1: [B, 3, H, W] -> [B, 64, H/2, W/2] (7x7 conv, stride=2)
            bn1, relu: [B, 64, H/2, W/2] (no shape change)
            maxpool: [B, 64, H/2, W/2] -> [B, 64, H/4, W/4] (3x3 pool, stride=2)
            layer1: [B, 64, H/4, W/4] -> [B, 256, H/4, W/4] (no downsampling)
            layer2: [B, 256, H/4, W/4] -> [B, 512, H/8, W/8] (stride=2 or dilation)
            layer3: [B, 512, H/8, W/8] -> [B, 1024, H/16, W/16] (stride=2 or dilation)
            layer4: [B, 1024, H/16, W/16] -> [B, 2048, H/16 or H/8, W/16 or W/8] (depends on output_stride)
        """
        x = self.conv1(x)  # [B, 3, H, W] -> [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, H/2, W/2] -> [B, 64, H/4, W/4]

        x = self.layer1(x)  # [B, 64, H/4, W/4] -> [B, 256, H/4, W/4]
        x = self.layer2(x)  # [B, 256, H/4, W/4] -> [B, 512, H/8, W/8] (or H/4 with dilation)
        x = self.layer3(x)  # [B, 512, H/8, W/8] -> [B, 1024, H/16, W/16] (or H/8 with dilation)
        x = self.layer4(x)  # [B, 1024, H/16, W/16] -> [B, 2048, H/16 or H/8, W/16 or W/8]

        return x


def _resnet(block, layers, **kwargs):
    """
    Helper function to create ResNet models.
    
    Args:
        block: The residual block type
        layers: Number of blocks in each layer
        **kwargs: Additional arguments for ResNet initialization
        
    Returns:
        ResNet model instance
    """
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    
    ResNet-50 has [3, 4, 6, 3] blocks in its four layers respectively.
    
    Args:
        **kwargs: Additional arguments for ResNet initialization
        
    Returns:
        ResNet-50 model instance
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    
    ResNet-101 has [3, 4, 23, 3] blocks in its four layers respectively.
    
    Args:
        **kwargs: Additional arguments for ResNet initialization
        
    Returns:
        ResNet-101 model instance
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


# ============================================
# Backbone: MobileNetV2
# ============================================

def _make_divisible(v, divisor, min_value=None):
    """
    Ensures that a value is divisible by a given divisor.
    
    This function is used to adjust channel numbers to be compatible
    with hardware optimizations.
    
    Args:
        v: Original value
        divisor: Divisor to make v divisible by
        min_value: Minimum value to return (default: divisor)
        
    Returns:
        The closest value >= v that is divisible by divisor
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    A convenience module that combines convolution, batch normalization, and ReLU6.
    
    This is a fundamental building block in MobileNetV2.
    
    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, 0, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def fixed_padding(kernel_size, dilation):
    """
    Calculates padding values for a given kernel size and dilation.
    
    This ensures that the convolution output has the same spatial dimensions
    as the input when stride=1.
    
    Args:
        kernel_size: Size of the convolution kernel
        dilation: Dilation rate
        
    Returns:
        Tuple of (pad_left, pad_right, pad_top, pad_bottom)
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class InvertedResidual(nn.Module):
    """
    Inverted residual block for MobileNetV2.
    
    This block features an inverted bottleneck design where we first expand
    the channel dimensions, apply a depth-wise convolution, and then project
    back to a lower dimension. A residual connection is used when stride=1
    and input/output channels match.
    
    Args:
        inp: Number of input channels
        oup: Number of output channels
        stride: Convolution stride
        dilation: Dilation rate
        expand_ratio: Channel expansion ratio
    """
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.input_padding = fixed_padding(3, dilation)

    def forward(self, x):
        """
        Forward pass through the inverted residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying the inverted residual transformation
            
        Tensor Shape Changes:
            x: [B, C_in, H, W]
            x_pad: [B, C_in, H+2*pad, W+2*pad] (after padding for dilation)
            If expand_ratio != 1:
                1x1 conv: [B, C_in, ...] -> [B, hidden_dim, ...] (channel expansion)
            3x3 depthwise conv: [B, hidden_dim, ...] -> [B, hidden_dim, ...'] (may downsample if stride>1)
            1x1 conv: [B, hidden_dim, ...'] -> [B, C_out, ...'] (channel projection)
            If use_res_connect: output = x + conv(x_pad) (element-wise addition)
        """
        x_pad = F.pad(x, self.input_padding)  # Apply padding for dilation: [B, C_in, H, W] -> [B, C_in, H+pad, W+pad]
        if self.use_res_connect:
            return x + self.conv(x_pad)  # Residual connection: [B, C, H, W] + [B, C, H, W]
        else:
            return self.conv(x_pad)  # Direct pass: [B, C_in, ...] -> [B, C_out, ...']


class MobileNetV2(nn.Module):
    """
    MobileNetV2 backbone network.
    
    MobileNetV2 uses inverted residuals and linear bottlenecks to achieve
    efficient computation, making it suitable for mobile and embedded applications.
    
    Args:
        output_stride: Desired output stride of the network
        width_mult: Channel width multiplier to scale the network
        inverted_residual_setting: Custom inverted residual block settings
    """
    def __init__(self, output_stride=8, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.output_stride = output_stride
        current_stride = 1
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, 8)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        current_stride *= 2
        dilation = 1
        previous_dilation = 1

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            previous_dilation = dilation
            if current_stride == output_stride:
                stride = 1
                dilation *= s
            else:
                stride = s
                current_stride *= s
            output_channel = int(c * width_mult)

            for i in range(n):
                if i == 0:
                    features.append(block(input_channel, output_channel, stride, previous_dilation, expand_ratio=t))
                else:
                    features.append(block(input_channel, output_channel, 1, dilation, expand_ratio=t))
                input_channel = output_channel
        
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through MobileNetV2.
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            Output features from the final layer
        """
        return self.features(x)


def mobilenet_v2(**kwargs):
    """
    Constructs a MobileNetV2 model.
    
    Args:
        **kwargs: Additional arguments for MobileNetV2 initialization
        
    Returns:
        MobileNetV2 model instance
    """
    return MobileNetV2(**kwargs)


# ============================================
# ASPP Module
# ============================================

class ASPPConv(nn.Sequential):
    """
    Atrous Convolution module for ASPP.
    
    Performs a 3x3 atrous convolution followed by batch normalization and ReLU.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate for the atrous convolution
    """
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    """
    Image pooling module for ASPP.
    
    Performs adaptive average pooling to 1x1, applies a 1x1 convolution,
    and then interpolates back to the original spatial size. This helps
    capture global contextual information.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Forward pass through the ASPP pooling module.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with global pooling features, interpolated
            to match the input spatial dimensions
            
        Tensor Shape Changes:
            x: [B, C_in, H, W]
            AdaptiveAvgPool2d(1): [B, C_in, H, W] -> [B, C_in, 1, 1] (global average pooling)
            1x1 conv: [B, C_in, 1, 1] -> [B, C_out, 1, 1]
            BN + ReLU: [B, C_out, 1, 1] (no shape change)
            Bilinear interpolate: [B, C_out, 1, 1] -> [B, C_out, H, W] (upsample to original size)
        """
        size = x.shape[-2:]  # Save original spatial dimensions: (H, W)
        x = super(ASPPPooling, self).forward(x)  # Global pool + 1x1 conv: [B, C_in, H, W] -> [B, C_out, 1, 1]
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # Upsample: [B, C_out, 1, 1] -> [B, C_out, H, W]


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    
    ASPP applies multiple parallel atrous convolutions with different
    dilation rates to capture multi-scale contextual information. It also
    includes an image-level pooling branch to capture global context.
    
    Args:
        in_channels: Number of input channels
        atrous_rates: List of dilation rates for the parallel atrous convolutions
    """
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        """
        Forward pass through ASPP.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Multi-scale feature tensor after ASPP processing
            
        Tensor Shape Changes:
            x: [B, C_in, H, W]
            
            Branch 0 (1x1 conv):
                [B, C_in, H, W] -> [B, 256, H, W]
            
            Branch 1-3 (3x3 atrous conv with different rates):
                [B, C_in, H, W] -> [B, 256, H, W] (each branch, maintains spatial size)
            
            Branch 4 (Image pooling):
                [B, C_in, H, W] -> [B, 256, 1, 1] -> [B, 256, H, W] (pool + upsample)
            
            Concatenation:
                5 branches: [B, 256, H, W] * 5 -> [B, 1280, H, W]
            
            Projection:
                [B, 1280, H, W] -> [B, 256, H, W] (1x1 conv + BN + ReLU + Dropout)
        """
        res = []
        for conv in self.convs:
            res.append(conv(x))  # Each branch: [B, C_in, H, W] -> [B, 256, H, W]
        res = torch.cat(res, dim=1)  # Concatenate 5 branches: [B, 256*5, H, W] = [B, 1280, H, W]
        return self.project(res)  # Projection: [B, 1280, H, W] -> [B, 256, H, W]


# ============================================
# DeepLab Head V3+
# ============================================

class DeepLabHeadV3Plus(nn.Module):
    """
    DeepLabV3+ decoder head.
    
    This head combines:
    1. Low-level features from the backbone (for spatial details)
    2. Multi-scale features from ASPP (for semantic information)
    
    The low-level features are projected to a smaller dimension, and the
    ASPP output is upsampled and concatenated with the low-level features
    before final classification.
    
    Args:
        in_channels: Number of channels from the high-level features
        low_level_channels: Number of channels from the low-level features
        num_classes: Number of segmentation classes
        aspp_dilate: List of dilation rates for ASPP
    """
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        # Feature extractor part (before final 1x1 conv for classification)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Classification head (final 1x1 conv)
        self.classification_head = nn.Conv2d(256, num_classes, 1)
        
        self._init_weight()

    def forward(self, feature, return_features=False):
        """
        Forward pass through the DeepLabV3+ head.
        
        Args:
            feature: Dictionary containing 'low_level' and 'out' features
            return_features: If True, also return the intermediate features before classification
            
        Returns:
            If return_features=False:
                Segmentation logits tensor [B, num_classes, H/4, W/4]
            If return_features=True:
                Tuple of (logits, features) where features are [B, 256, H/4, W/4]
            
        Tensor Shape Changes:
            feature['low_level']: [B, 256, H/4, W/4] (from layer1 of ResNet)
            feature['out']: [B, 2048, H/16 or H/8, W/16 or W/8] (from layer4 of ResNet)
            
            Low-level projection:
                [B, 256, H/4, W/4] -> [B, 48, H/4, W/4] (1x1 conv for dimensionality reduction)
            
            ASPP processing:
                [B, 2048, H/16, W/16] -> [B, 256, H/16, W/16] (multi-scale feature extraction)
            
            Upsampling ASPP output:
                [B, 256, H/16, W/16] -> [B, 256, H/4, W/4] (4x upsampling to match low-level features)
            
            Concatenation:
                [B, 48, H/4, W/4] + [B, 256, H/4, W/4] -> [B, 304, H/4, W/4]
            
            Feature Extractor:
                [B, 304, H/4, W/4] -> [B, 256, H/4, W/4] (3x3 conv, this is the geometric feature output)
            
            Classification Head:
                [B, 256, H/4, W/4] -> [B, num_classes, H/4, W/4] (1x1 conv)
        """
        low_level_feature = self.project(feature['low_level'])  # [B, 256, H/4, W/4] -> [B, 48, H/4, W/4]
        output_feature = self.aspp(feature['out'])  # [B, 2048, H/16, W/16] -> [B, 256, H/16, W/16]
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)  # [B, 256, H/16, W/16] -> [B, 256, H/4, W/4]
        
        # Extract features (this is what we want for the geometric branch)
        features = self.feature_extractor(torch.cat([low_level_feature, output_feature], dim=1))  # [B, 304, H/4, W/4] -> [B, 256, H/4, W/4]
        
        # Classification
        logits = self.classification_head(features)  # [B, 256, H/4, W/4] -> [B, num_classes, H/4, W/4]
        
        if return_features:
            return logits, features
        return logits

    def _init_weight(self):
        """
        Initialize weights for the classifier modules.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ============================================
# DeepLabV3+ Model
# ============================================

class DeepLabV3(_SimpleSegmentationModel):
    """
    DeepLabV3+ model for semantic segmentation.
    
    This class wraps the backbone and classifier into a single segmentation model
    that inherits from _SimpleSegmentationModel.
    
    Extended with methods for:
    - Head replacement for transfer learning
    - Feature extraction for geometric branch
    """
    
    def replace_classification_head(self, num_classes):
        """
        Replace the classification head with a new randomly initialized one.
        
        This is useful for transfer learning to different datasets with
        different number of classes (e.g., from VOC 21 classes to
        2 classes for keypoint detection: main scale and pointer).
        
        Args:
            num_classes: Number of output classes for the new head
            
        Example:
            >>> model = deeplabv3plus_resnet50(num_classes=21)
            >>> model = load_pretrained_weights(model, 'checkpoints/pretrained.pth')
            >>> model.replace_classification_head(num_classes=2)  # Now for 2-class keypoint detection
        """
        # Get the in_channels from the existing classification head
        in_channels = self.classifier.classification_head.in_channels
        
        # Create new classification head with random initialization
        self.classifier.classification_head = nn.Conv2d(in_channels, num_classes, 1)
        
        # Initialize the new head
        nn.init.kaiming_normal_(self.classifier.classification_head.weight)
        if self.classifier.classification_head.bias is not None:
            nn.init.zeros_(self.classifier.classification_head.bias)
    
    def extract_features(self, x):
        """
        Extract geometric features (before the final classification head).
        
        This returns the [B, 256, H/4, W/4] feature map which is
        used as the output of the geometric branch in the GaugeOCR model.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            
        Returns:
            features: Geometric feature tensor of shape [B, 256, H/4, W/4]
            
        Example:
            >>> model = deeplabv3plus_resnet50(num_classes=21)
            >>> x = torch.randn(1, 3, 448, 448)
            >>> features = model.extract_features(x)
            >>> print(features.shape)  # torch.Size([1, 256, 112, 112])
        """
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        _, geometric_features = self.classifier(features, return_features=True)
        return geometric_features
    
    def forward(self, x, return_features=False):
        """
        Forward pass through the DeepLabV3+ model.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            return_features: If True, also return the geometric features
            
        Returns:
            If return_features=False:
                Segmentation logits tensor of shape [B, num_classes, H, W]
            If return_features=True:
                Tuple of (logits, features) where features are [B, 256, H/4, W/4]
        """
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        
        if return_features:
            logits, geometric_features = self.classifier(features, return_features=True)
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
            return logits, geometric_features
        else:
            logits = self.classifier(features)
            logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
            return logits


# ============================================
# Model Constructors
# ============================================

def _segm_resnet(name, backbone_name, num_classes, output_stride):
    """
    Helper function to create DeepLabV3+ with ResNet backbone.
    
    Args:
        name: Model name
        backbone_name: 'resnet50' or 'resnet101'
        num_classes: Number of segmentation classes
        output_stride: Desired output stride (8 or 16)
        
    Returns:
        DeepLabV3+ model with ResNet backbone
    """
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name == 'resnet50':
        backbone = resnet50(replace_stride_with_dilation=replace_stride_with_dilation)
    elif backbone_name == 'resnet101':
        backbone = resnet101(replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride):
    """
    Helper function to create DeepLabV3+ with MobileNetV2 backbone.
    
    Args:
        name: Model name
        backbone_name: 'mobilenetv2'
        num_classes: Number of segmentation classes
        output_stride: Desired output stride (8 or 16)
        
    Returns:
        DeepLabV3+ model with MobileNetV2 backbone
    """
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenet_v2(output_stride=output_stride)
    
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None

    inplanes = 320
    low_level_planes = 24

    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def deeplabv3plus_resnet50(num_classes=21, output_stride=16):
    """
    Constructs DeepLabV3+ with ResNet50 backbone.
    
    Args:
        num_classes: Number of segmentation classes (default: 21 for PASCAL VOC)
        output_stride: Output stride, either 8 or 16 (default: 16)
        
    Returns:
        DeepLabV3+ model with ResNet50 backbone
    """
    return _segm_resnet('deeplabv3plus', 'resnet50', num_classes, output_stride)


def deeplabv3plus_resnet101(num_classes=21, output_stride=16):
    """
    Constructs DeepLabV3+ with ResNet101 backbone.
    
    Args:
        num_classes: Number of segmentation classes (default: 21 for PASCAL VOC)
        output_stride: Output stride, either 8 or 16 (default: 16)
        
    Returns:
        DeepLabV3+ model with ResNet101 backbone
    """
    return _segm_resnet('deeplabv3plus', 'resnet101', num_classes, output_stride)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=16):
    """
    Constructs DeepLabV3+ with MobileNetV2 backbone.
    
    Args:
        num_classes: Number of segmentation classes (default: 21 for PASCAL VOC)
        output_stride: Output stride, either 8 or 16 (default: 16)
        
    Returns:
        DeepLabV3+ model with MobileNetV2 backbone
    """
    return _segm_mobilenet('deeplabv3plus', 'mobilenetv2', num_classes, output_stride)


# ============================================
# HuggingFace-style Model Creation API
# ============================================

DEFAULT_PRETRAINED_PATH = "checkpoints/deeplabv3plus_resnet50.pth"


def create_deeplabv3plus(backbone='resnet50', num_classes=21, output_stride=16,
                         pretrained=True, pretrained_path=None, 
                         weights_only=True):
    """
    Create DeepLabV3+ model - HuggingFace-style API.
    
    This function provides a convenient way to create DeepLabV3+ models
    with automatic pretrained weight loading and head replacement.
    
    Usage Examples:
        # 1. Create model with 21 classes (for loading pretrained weights)
        model = create_deeplabv3plus(backbone='resnet50', num_classes=21, pretrained=True)
        
        # 2. Create model with custom classes (automatically loads pretrained and replaces head)
        model = create_deeplabv3plus(backbone='resnet50', num_classes=3, pretrained=True)
        
        # 3. Create model without pretrained weights
        model = create_deeplabv3plus(backbone='resnet50', num_classes=3, pretrained=False)
        
        # 4. Specify custom pretrained weights path
        model = create_deeplabv3plus(backbone='resnet50', num_classes=3, 
                                      pretrained=True, pretrained_path='custom.pth')
    
    Args:
        backbone: Backbone architecture ('resnet50', 'resnet101', or 'mobilenet')
        num_classes: Number of segmentation classes
        output_stride: Output stride (8 or 16)
        pretrained: Whether to load pretrained weights
        pretrained_path: Custom path to pretrained weights (optional)
        weights_only: Whether to use weights_only=True when loading (for security)
    
    Returns:
        DeepLabV3+ model with specified configuration
    
    Note:
        When pretrained=True and num_classes != 21:
        1. First creates model with 21 classes to match pretrained weights
        2. Loads pretrained weights (PASCAL VOC 21 classes)
        3. Replaces classification head with new randomly initialized head for num_classes
    """
    # Validate backbone
    backbone = backbone.lower()
    if backbone not in ['resnet50', 'resnet101', 'mobilenet']:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from 'resnet50', 'resnet101', 'mobilenet'")
    
    # Determine pretrained path
    if pretrained and pretrained_path is None:
        # Use default path - user should provide the actual pretrained weights
        pretrained_path = DEFAULT_PRETRAINED_PATH
    
    # Create model
    if backbone == 'resnet50':
        model = deeplabv3plus_resnet50(num_classes=21, output_stride=output_stride)
    elif backbone == 'resnet101':
        model = deeplabv3plus_resnet101(num_classes=21, output_stride=output_stride)
    else:  # mobilenet
        model = deeplabv3plus_mobilenet(num_classes=21, output_stride=output_stride)
    
    # Load pretrained weights and/or replace head
    if pretrained and pretrained_path and os.path.exists(pretrained_path):
        print(f"[create_deeplabv3plus] Loading pretrained weights from: {pretrained_path}")
        try:
            state_dict = load_pretrained_weights(pretrained_path, weights_only=weights_only)
            
            # Handle different key formats
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                new_state_dict[k] = v
            
            # Try to load pretrained weights
            model_dict = model.state_dict()
            loaded_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(loaded_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"[create_deeplabv3plus] Loaded {len(loaded_dict)}/{len(model_dict)} layers from pretrained")
            
        except Exception as e:
            print(f"[create_deeplabv3plus] Warning: Failed to load pretrained weights: {e}")
            print(f"[create_deeplabv3plus] Continuing with randomly initialized weights")
    
    # Replace classification head if num_classes != 21
    if num_classes != 21:
        print(f"[create_deeplabv3plus] Replacing classification head: 21 -> {num_classes} classes")
        model.replace_classification_head(num_classes)
    
    return model


# ============================================
# Loading Pretrained Weights
# ============================================

def load_pretrained_weights(model, checkpoint_path):
    """
    Load pretrained weights into a DeepLabV3+ model.
    Handles both old and new model structures.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取 state_dict
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    # 创建新的 state_dict 用于加载
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # 处理旧结构到新结构的映射
        if key.startswith('classifier.classifier.'):
            # 旧的 classifier.classifier.X → 新的 classifier.feature_extractor.X
            if 'classifier.classifier.3' in key:  # 最后的分类层
                # 映射到 classification_head
                new_key = key.replace('classifier.classifier.3', 'classifier.classification_head')
            elif 'classifier.classifier.0' in key or 'classifier.classifier.1' in key or 'classifier.classifier.2' in key:
                # 映射到 feature_extractor
                new_key = key.replace('classifier.classifier', 'classifier.feature_extractor')
            else:
                new_key = key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # 尝试加载
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded weights with strict=True")
    except RuntimeError as e:
        print(f"Strict loading failed, trying non-strict...")
        # 如果严格加载失败，使用 non-strict 加载
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys[:5]}...")  # 只显示前5个
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
        
        print("Loaded weights with strict=False (ignoring mismatched keys)")
    
    return model


# ============================================
# Output Processing for VOC 2012
# ============================================

# PASCAL VOC 2012 color map for visualization
# Each class has a unique RGB color for easy visualization
VOC_COLOR_MAP = [
    [0, 0, 0],         # 0: background
    [128, 0, 0],       # 1: aeroplane
    [0, 128, 0],       # 2: bicycle
    [128, 128, 0],     # 3: bird
    [0, 0, 128],       # 4: boat
    [128, 0, 128],     # 5: bottle
    [0, 128, 128],     # 6: bus
    [128, 128, 128],   # 7: car
    [64, 0, 0],        # 8: cat
    [192, 0, 0],       # 9: chair
    [64, 128, 0],      # 10: cow
    [192, 128, 0],     # 11: diningtable
    [64, 0, 128],      # 12: dog
    [192, 0, 128],     # 13: horse
    [64, 128, 128],    # 14: motorbike
    [192, 128, 128],   # 15: person
    [0, 64, 0],        # 16: pottedplant
    [128, 64, 0],      # 17: sheep
    [0, 192, 0],       # 18: sofa
    [128, 192, 0],     # 19: train
    [0, 64, 128],      # 20: tvmonitor
]

# PASCAL VOC 2012 class names
VOC_CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def decode_predictions(logits):
    """
    Convert model logits to segmentation predictions.
    
    This function takes the raw output logits from the model and converts them
    to class predictions by taking the argmax along the class dimension.
    
    Args:
        logits: Model output tensor of shape [B, num_classes, H, W]
               where B is batch size, num_classes is 21 for VOC 2012
        
    Returns:
        predictions: Tensor of shape [B, H, W] containing class indices (0-20)
        probabilities: Tensor of shape [B, H, W] containing max probability values
    """
    # Get class predictions by taking argmax along class dimension
    # logits: [B, num_classes, H, W] -> predictions: [B, H, W]
    probabilities, predictions = torch.max(logits, dim=1)
    
    return predictions, probabilities


def logits_to_color_mask(logits, color_map=VOC_COLOR_MAP):
    """
    Convert model logits to a color-coded segmentation mask.
    
    This function converts the raw model output to a color image where each
    semantic class is represented by a unique color, making it easy to visualize
    the segmentation results.
    
    Args:
        logits: Model output tensor of shape [B, num_classes, H, W]
               Can be on CPU or GPU, will be converted to numpy array
        color_map: List of RGB colors for each class (default: VOC_COLOR_MAP)
        
    Returns:
        color_masks: List of PIL Image objects in RGB mode, one for each batch
                    Each image has shape [H, W, 3] with uint8 values
    """
    # Get predictions: [B, H, W]
    predictions, _ = decode_predictions(logits)
    
    # Move to CPU and convert to numpy
    predictions = predictions.cpu().numpy()
    
    # Create color map array: [num_classes, 3]
    color_map_array = np.array(color_map, dtype=np.uint8)
    
    color_masks = []
    for i in range(predictions.shape[0]):
        # Map each pixel's class index to its RGB color
        # predictions[i]: [H, W] -> pred_rgb: [H, W, 3]
        pred_rgb = color_map_array[predictions[i]]
        
        # Convert to PIL Image
        color_mask = Image.fromarray(pred_rgb, mode='RGB')
        color_masks.append(color_mask)
    
    return color_masks


def save_segmentation_results(logits, image_paths, output_dir, save_color=True, save_mask=True):
    """
    Save segmentation predictions to disk.
    
    This function saves both the color-coded segmentation masks and the raw
    class index masks to disk for evaluation or visualization purposes.
    
    Args:
        logits: Model output tensor of shape [B, num_classes, H, W]
        image_paths: List of input image paths (used for naming output files)
        output_dir: Directory to save the output files
        save_color: If True, save color-coded segmentation masks (default: True)
        save_mask: If True, save raw class index masks as PNG (default: True)
        
    Returns:
        saved_files: List of paths to saved files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions: [B, H, W]
    predictions, _ = decode_predictions(logits)
    predictions = predictions.cpu().numpy()
    
    saved_files = []
    
    for i in range(len(image_paths)):
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
        
        if save_color:
            # Save color-coded mask
            color_masks = logits_to_color_mask(logits[i:i+1])
            color_path = os.path.join(output_dir, f'{base_name}_color.png')
            color_masks[0].save(color_path)
            saved_files.append(color_path)
        
        if save_mask:
            # Save raw class index mask (single channel, uint8)
            mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
            mask = Image.fromarray(predictions[i].astype(np.uint8), mode='L')
            mask.save(mask_path)
            saved_files.append(mask_path)
    
    return saved_files


def compute_iou(predictions, ground_truth, num_classes=21):
    """
    Compute Intersection over Union (IoU) for each class.
    
    IoU is the standard evaluation metric for semantic segmentation.
    It measures the overlap between predicted and ground truth regions.
    
    Args:
        predictions: Predicted segmentation mask of shape [H, W] or [B, H, W]
                    containing class indices (0 to num_classes-1)
        ground_truth: Ground truth mask of shape [H, W] or [B, H, W]
                     containing class indices (0 to num_classes-1)
        num_classes: Number of semantic classes (default: 21 for VOC 2012)
        
    Returns:
        iou_per_class: numpy array of shape [num_classes] containing IoU for each class
        mean_iou: Mean IoU across all classes (scalar)
    """
    # Ensure inputs are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    
    # Handle batch dimension
    if predictions.ndim == 3:
        predictions = predictions[0]
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[0]
    
    # Initialize IoU for each class
    iou_per_class = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # Create binary masks for current class
        pred_mask = (predictions == cls)
        gt_mask = (ground_truth == cls)
        
        # Compute intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # Compute IoU (handle division by zero)
        if union == 0:
            iou_per_class[cls] = np.nan  # Class not present in ground truth
        else:
            iou_per_class[cls] = intersection / union
    
    # Compute mean IoU (ignore NaN values)
    mean_iou = np.nanmean(iou_per_class)
    
    return iou_per_class, mean_iou


def visualize_segmentation(image, logits, class_names=VOC_CLASS_NAMES, alpha=0.5):
    """
    Create a visualization overlay of segmentation results on the original image.
    
    This function blends the segmentation mask with the original image to create
    an intuitive visualization of the model's predictions.
    
    Args:
        image: Input image as PIL Image or numpy array of shape [H, W, 3]
        logits: Model output tensor of shape [1, num_classes, H, W]
               Note: H and W should match the image dimensions
        class_names: List of class names for legend (default: VOC_CLASS_NAMES)
        alpha: Blending factor between 0 and 1 (default: 0.5)
              0 = only original image, 1 = only segmentation mask
        
    Returns:
        overlay: PIL Image showing the blended segmentation result
    """
    # Convert image to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Get color mask
    color_masks = logits_to_color_mask(logits)
    color_mask_np = np.array(color_masks[0])
    
    # Blend the image and color mask
    # overlay = image * (1 - alpha) + color_mask * alpha
    overlay_np = (image_np * (1 - alpha) + color_mask_np * alpha).astype(np.uint8)
    
    # Convert back to PIL Image
    overlay = Image.fromarray(overlay_np)
    
    return overlay


# ============================================
# Usage Example
# ============================================

def pil_to_tensor(image_path):
    """
    将图像转换为模型输入格式 (1, 3, H, W)
    
    Args:
        image_path: 图像路径
        
    Returns:
        torch.Tensor: 形状为 (1, 3, H, W) 的归一化张量
        tuple: 原始图像尺寸 (width, height)
    """
    import torchvision.transforms as transforms
    # 1. 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 2. 定义转换流程
    transform = transforms.Compose([
        transforms.ToTensor(),           # PIL -> Tensor, 自动归一化到 [0,1]
        transforms.Resize((448, 448)),
        transforms.Normalize(            # ImageNet 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 3. 转换并添加 batch 维度
    tensor = transform(image).unsqueeze(0)  # (3, H, W) -> (1, 3, H, W)

    tensor_size = tensor.shape
    
    return tensor, original_size, tensor_size


if __name__ == '__main__':
    # Example 1: Create DeepLabV3+ with ResNet50 backbone
    print("=" * 60)
    print("Creating DeepLabV3+ (ResNet50 backbone)")
    model_resnet50 = deeplabv3plus_resnet50(num_classes=21, output_stride=16)
    model_resnet50 = load_pretrained_weights(model_resnet50, 'best_deeplabv3plus_resnet50_voc_os16.pth')
    print(f"Number of parameters: {sum(p.numel() for p in model_resnet50.parameters()) / 1e6:.2f}M")
    
    # Example 2: Forward pass test
    print("\n" + "=" * 60)
    print("Forward pass test")
    x, original_size, tensor_size = pil_to_tensor('0240.jpg_wh860.jpg')
    model_resnet50.eval()
    with torch.no_grad():
        output = model_resnet50(x)
        features = model_resnet50.extract_features(x)
    print(f"Features shape: {features.shape}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Example 3: Decode predictions
    print("\n" + "=" * 60)
    print("Decoding predictions")
    predictions, probabilities = decode_predictions(output)
    print(f"Predictions shape: {predictions.shape}")  # [B, H, W]
    print(f"Probabilities shape: {probabilities.shape}")  # [B, H, W]
    print(f"Unique classes in prediction: {torch.unique(predictions).tolist()}")
    
    # Example 4: Convert to color mask
    print("\n" + "=" * 60)
    print("Converting to color mask")
    color_masks = logits_to_color_mask(output)
    print(f"Number of masks: {len(color_masks)}")
    print(f"Mask size: {color_masks[0].size}")  # PIL Image size
    
    # Example 5: Save segmentation results
    print("\n" + "=" * 60)
    print("Saving segmentation results")
    import os
    output_dir = 'segmentation_results'
    os.makedirs(output_dir, exist_ok=True)
    image_paths = ['test_image.jpg']  # Example image path
    saved_files = save_segmentation_results(output, image_paths, output_dir)
    print(f"Saved files: {saved_files}")
    
    # Example 6: Compute IoU (with dummy ground truth for demonstration)
    print("\n" + "=" * 60)
    print("Computing IoU (with dummy ground truth)")
    dummy_gt = torch.zeros_like(predictions)  # All background for demo
    iou_per_class, mean_iou = compute_iou(predictions, dummy_gt)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"IoU per class (first 5): {iou_per_class[:5]}")
    
    # Example 7: Visualization (commented, requires actual image)
    print("\n" + "=" * 60)
    print("Visualization example (commented):")
    print("""
    # To visualize segmentation on an actual image:
    from PIL import Image
    
    # Load image
    image = Image.open('path/to/image.jpg')
    image = image.resize((448, 448))  # Match model input size
    
    # Preprocess and run model
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Create visualization
    overlay = visualize_segmentation(image, output, alpha=0.5)
    overlay.save('visualization.png')
    """)
    
    # Example 8: Print class names
    print("\n" + "=" * 60)
    print("PASCAL VOC 2012 Classes:")
    for i, name in enumerate(VOC_CLASS_NAMES):
        color = VOC_COLOR_MAP[i]
        print(f"  {i:2d}: {name:15s} - RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d})")
    
    print("\n" + "=" * 60)
    print("Complete!")
