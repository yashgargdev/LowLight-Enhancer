import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LowLightNet(nn.Module):
    """U-Net Architecture for V3 Upgrade (Better Color/Structure Retention)"""
    def __init__(self):
        super(LowLightNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)
        
        # Decoder (Upsampling & Skip Connections)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        # Final output layer
        self.out_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder with Skip Connections (Copying detailed edges from Encoder)
        # We pad the upsampled tensor to match the encoder tensor size
        # This handles cases where input dimensions aren't divisible by 8
        u3 = self.upconv3(b)
        diffY = e3.size()[2] - u3.size()[2]
        diffX = e3.size()[3] - u3.size()[3]
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u3 = torch.cat([u3, e3], dim=1) 
        d3 = self.dec3(u3)
        
        u2 = self.upconv2(d3)
        diffY = e2.size()[2] - u2.size()[2]
        diffX = e2.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        diffY = e1.size()[2] - u1.size()[2]
        diffX = e1.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        # Output
        out = self.out_conv(d1)
        return self.sigmoid(out)
