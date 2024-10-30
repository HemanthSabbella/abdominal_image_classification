import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def conv_block(in_c, out_c):
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            return block
        
        def upconv_block(in_c, out_c):
            block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )
            return block

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = upconv_block(1024, 512)
        self.conv_up4 = conv_block(1024, 512)
        
        self.upconv3 = upconv_block(512, 256)
        self.conv_up3 = conv_block(512, 256)

        # Define the auxiliary output layer here
        self.auxiliary_conv = nn.Conv2d(256, out_channels, kernel_size=1)  # Auxiliary output

        self.upconv2 = upconv_block(256, 128)
        self.conv_up2 = conv_block(256, 128)

        self.upconv1 = upconv_block(128, 64)
        self.conv_up1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        c5 = self.conv5(p4)

        u4 = self.upconv4(c5)
        u4 = torch.cat([u4, c4], dim=1)
        u4 = self.conv_up4(u4)

        u3 = self.upconv3(u4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.conv_up3(u3)

        # Generate auxiliary output
        aux_output = self.auxiliary_conv(u3)  # Auxiliary output after upsampling

        u2 = self.upconv2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.conv_up2(u2)

        u1 = self.upconv1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.conv_up1(u1)

        out = self.final_conv(u1)
        return out, aux_output # added output for axuiliary loss