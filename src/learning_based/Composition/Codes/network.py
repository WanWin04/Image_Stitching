import torch
import torch.nn as nn
import torch.nn.functional as F


# Build the model to generate learned masks and the stitched image
def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor):
    # Forward pass through the network
    out = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    # Compute learned masks based on the input masks and the network output
    learned_mask1 = (mask1_tensor - mask1_tensor * mask2_tensor) + mask1_tensor * mask2_tensor * out
    learned_mask2 = (mask2_tensor - mask1_tensor * mask2_tensor) + mask1_tensor * mask2_tensor * (1 - out)
    
    # Compute the stitched image using the learned masks and the warped input images
    stitched_image = (warp1_tensor + 1.) * learned_mask1 + (warp2_tensor + 1.) * learned_mask2 - 1.

    # Output dictionary containing the results
    out_dict = {}
    out_dict.update(learned_mask1=learned_mask1, learned_mask2=learned_mask2, stitched_image=stitched_image)

    return out_dict


# Down-sampling block for feature extraction
class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        # Optional max pooling for spatial reduction
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # Two convolutional layers with ReLU activation
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation=dilation))
        blk.append(nn.ReLU(inplace=True))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation=dilation))
        blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)


# Up-sampling block for reconstructing spatial details
class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        # Adjust input channels using a convolutional layer
        self.halfChanelConv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Two convolutional layers for refinement
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation=dilation),
            nn.ReLU(inplace=True)
        )

        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        # Upsample the lower resolution feature map
        x1 = F.interpolate(x1, size=(x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        # Concatenate features from both resolutions
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


# Network for predicting the composition mask and stitching
class Network(nn.Module):
    def __init__(self, nclasses=1):
        super(Network, self).__init__()

        # Down-sampling blocks for hierarchical feature extraction
        self.down1 = DownBlock(3, 32, 1, pool=False)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 128, 3)
        self.down4 = DownBlock(128, 256, 4)
        self.down5 = DownBlock(256, 512, 5)

        # Up-sampling blocks for reconstructing spatial details
        self.up1 = UpBlock(512, 256, 4)
        self.up2 = UpBlock(256, 128, 3)
        self.up3 = UpBlock(128, 64, 2)
        self.up4 = UpBlock(64, 32, 1)

        # Final layer to output the composition mask
        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1),
            nn.Sigmoid()
        )

        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, y, m1, m2):
        # Down-sampling paths for both input images
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.down4(y3)
        y5 = self.down5(y4)

        # Combine features using up-sampling and subtraction
        res = self.up1(x5 - y5, x4 - y4)
        res = self.up2(res, x3 - y3)
        res = self.up3(res, x2 - y2)
        res = self.up4(res, x1 - y1)

        # Output the predicted mask
        res = self.out(res)

        return res
