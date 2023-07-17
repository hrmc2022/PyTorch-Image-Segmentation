import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    '''Convlutional Block
    Args:
        in_ch (int): input channel
        mid_ch (int): middle channel
        out_ch (int): output channel
    '''
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch, track_running_stats=True)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class LastConvBlock(nn.Module):
    '''Convlutional Block
    Args:
        in_ch (int): input channel
        mid_ch (int): middle channel
        out_ch (int): output channel
    '''
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch, track_running_stats=True)
        
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UpSample(nn.Module):
    '''Upsampling Convlutional Block
    Args:
        in_ch (int): input channel
        mid_ch (int): middle channel
        out_ch (int): output channel
    '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding="same")
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_ch, track_running_stats=True)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        x = self.upsample(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    '''Upsampling Convlutional Block
    Args:
        in_ch (int): input channel
        out_ch (int): output channnel (= the number of classes)
    '''
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # self.down_conv1 = ConvBlock(in_ch, 64, 64)

        self.input_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.input_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.down_conv2 = ConvBlock(64, 128, 128)
        self.down_conv3 = ConvBlock(128, 256, 256)
        self.down_conv4 = ConvBlock(256, 512, 512)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.up_conv1 = ConvBlock(512, 1024, 1024)
        self.up_sample1 = UpSample(1024, 512)
        self.up_conv2 = ConvBlock(1024, 512, 512)
        self.up_sample2 = UpSample(512, 256)
        self.up_conv3 = ConvBlock(512, 256, 256)
        self.up_sample3 = UpSample(256, 128)
        self.up_conv4 = ConvBlock(256, 128, 128)
        self.up_sample4 = UpSample(128, 64)
        self.up_conv5 = ConvBlock(128, 64, 64)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
        self.last_conv = LastConvBlock(128, 64, 64)
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1, padding="same")
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor):
        # _, _, in_h, in_w = x.shape

        # Down sampling
        # x1 = self.down_conv1(x)
        # print(x.shape)
        x = self.relu(self.input_conv1(x))
        x1 = self.relu(self.input_conv2(x))
        x = self.maxpool(x1)
        x2 = self.down_conv2(x)
        x = self.maxpool(x2)
        x3 = self.down_conv3(x)
        x = self.maxpool(x3)
        x4 = self.down_conv4(x)
        x = self.maxpool(x4)

        # Up sampling
        # x = self.up_conv1(x)
        # x = torch.cat([x, self.crop(x4, x)], dim=1)
        # x = self.up_conv2(x)
        # x = torch.cat([x, self.crop(x3, x)], dim=1)
        # x = self.up_conv3(x)
        # x = torch.cat([x, self.crop(x2, x)], dim=1)
        # x = self.up_conv4(x)
        # x = torch.cat([x, self.crop(x1, x)], dim=1)
        # print("x4 ;", x4.shape)
        x = self.up_conv1(x)
        x = self.up_sample1(x)
        # print("up1: ", x.shape)
        x = torch.cat([x, x4], dim=1)
        # print("x: ", x.shape)
        x = self.up_conv2(x)
        x = self.up_sample2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv3(x)
        x = self.up_sample3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv4(x)
        x = self.up_sample4(x)
        x = torch.cat([x, x1], dim=1)

        # Final conv
        x = self.last_conv(x)
        x = self.out_conv(x)
        # x = self.soft(x)

        return x

    def crop(self, input, x):
        _, _, H, W = x.shape
        return torchvision.transforms.CenterCrop([H, W])(input)


if __name__ == "__main__":
    print("ConvBlock test")
    input = torch.randn(1, 512, 512).unsqueeze(dim=0)
    print(input.shape)
    out = ConvBlock(1, 64, 64)(input)    
    print(out.shape)

    # print("UpSample test")
    # input = torch.randn(128, 196, 196).unsqueeze(dim=0)
    # print(input.shape)
    # out = UpSample(128)(input)    
    # print(out.shape)

    print("UNet test")
    input = torch.randn(1, 512, 512).unsqueeze(dim=0)
    print(input.shape)
    out = UNet(1)(input)    
    print(out.shape)
