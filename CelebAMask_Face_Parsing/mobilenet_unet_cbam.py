import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block --> Reduces the no. of params while preserving spatial info
       Depthwise - applies a seperate convolution to each channel
       Pointwise - mixes channel information
       BatchNorm and ReLU - Stability and non-linearity"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, negative_slope=0.01):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)  # LeakyReLU activation


    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.leaky_relu(x)
    
class EncoderBlock(nn.Module):
    """Encoder block with depthwise separable conv + downsampling
       Depthwise separable convolution - extracts features
       Max-pooling - reduces spatial size by half
       Skip connections - returns original feature map and downsampled feature map"""
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.downsample = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) ## new
        x_down = self.downsample(x)
        return x, x_down  # Return both for skip connection
    
class DecoderBlock(nn.Module):
    """Decoder block with upsampling
       Transposed Convolution - upsampling (doubles spatial size) ORRR Bilinear upsampling
       Skip connection - combines high resolution encoder features
       Depthwise Seperable Convolution - refines features"""
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        # self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.conv1 = DepthwiseSeparableConv(out_channels * 2, out_channels)  # Concatenation doubles channels
        self.conv1 = DepthwiseSeparableConv(in_channels + out_channels, out_channels)  # Fix channel mismatch
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.conv1(x)
        x = self.conv2(x) 
        return x

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction=16, kernel_size=9):
        super().__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel Attention
        avg_out = self.mlp(self.avg_pool(x).view(B, C)).view(B, C, 1, 1)
        max_out = self.mlp(self.max_pool(x).view(B, C)).view(B, C, 1, 1)
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_att  # Apply channel attention

        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_pool, max_pool], dim=1)))
        x = x * spatial_att  # Apply spatial attention
        
        return x

class BottleneckWithCBAM(nn.Module):
    """Bottleneck block with CBAM attention"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.cbam(x)  # Apply CBAM instead of Self-Attention

class MobileNet_UNet_CBAM(nn.Module):
    """Lightweight U-Net with MobileNetV2-style depthwise separable convolutions + CBAM"""
    def __init__(self, num_classes=19):
        super().__init__()

        # Encoder Blocks
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # Bottleneck with CBAM
        self.bottleneck = BottleneckWithCBAM(256, 512)

        # Decoder Blocks
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        # Final 1x1 convolution for segmentation output
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)  # Apply CBAM in bottleneck

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.final_conv(x)
