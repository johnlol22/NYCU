import torch
import torch.nn as nn


class encoder_block(nn.Module):
    """Basic residual block for ResNet34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(encoder_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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


class ConvBlock(nn.Module):
    """Double convolution block for decoder"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        
        # Upsampling method: bilinear or transposed convolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample the input
        x1 = self.up(x1)
        
        # Handle size mismatches between encoder and decoder features
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                   diffY // 2, diffY - diffY // 2])
        
        # Concatenate the upsampled features with the skip connection features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ResNet34Encoder(nn.Module):
    """ResNet34 encoder implemented from scratch"""
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        
        # Initial convolution layer
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self.contraction(64, 3, stride=1)
        self.layer2 = self.contraction(128, 4, stride=2)
        self.layer3 = self.contraction(256, 6, stride=2)
        self.layer4 = self.contraction(512, 3, stride=2)
        
        # additional bottleneck ?
        self.layer5 = ConvBlock(512, 256)
        
        # Initialize weights
        self._initialize_weights()
        
    def contraction(self, out_channels, blocks, stride=1):
        downsample = None
        
        # Create downsample if stride != 1 or if channel dimensions change
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # First block with potential stride and downsample
        layers.append(encoder_block(self.in_channels, out_channels, stride, downsample))
        
        # Update input channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(encoder_block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Return intermediate outputs for skip connections
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)  # 1/4 resolution
        
        x1 = self.layer1(x0)   # 1/4 resolution, 64 channels
        x2 = self.layer2(x1)   # 1/8 resolution, 128 channels
        x3 = self.layer3(x2)   # 1/16 resolution, 256 channels
        x4 = self.layer4(x3)   # 1/32 resolution, 512 channels

        x5 = self.layer5(x4)
        
        return x0, x1, x2, x3, x4, x5


class ResUNet(nn.Module):
    """
    ResNet34 + UNet hybrid architecture for image segmentation,
    implemented from scratch without pre-defined models
    """
    def __init__(self, num_classes=1):
        super(ResUNet, self).__init__()
        
        # Custom ResNet34 encoder
        self.encoder = ResNet34Encoder()
        # Define decoder blocks
        self.decoder1 = Up(512 + 256, 32)
        self.decoder2 = Up(256 + 32, 32)
        self.decoder3 = Up(128 + 32, 32)
        self.decoder4 = Up(32 + 64, 32)
        
        self.decoder5 = ConvBlock(32, 32)
        
        # Upsampling to input size (since input is downsampled by 4 initially)
        self.upscale = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        # Final output convolution
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x0, x1, x2, x3, x4, x5= self.encoder(x)
        
        # Decoder forward pass with skip connections
        d1 = self.decoder1(x5, x4)
        d2 = self.decoder2(d1, x3)
        d3 = self.decoder3(d2, x2)
        d4 = self.decoder4(d3, x1)
        
        d5 = self.decoder5(d4)
        
        # Upscale back to original input size
        out = self.upscale(d5)
        
        # Final convolution
        out = self.outc(out)
        
        return out
