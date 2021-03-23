from torch import nn
import torch


class Lenet(nn.Module):
    def __init__(self, classification=2):
        super().__init__()
        self.net = nn.Sequential(
            # nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),  # Para: 150
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # Para: 2400
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1),  # Para: 25600
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10),  # Para: 640
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=classification),  # Para: 20
            nn.ReLU()
        )  # Total para: 28,810

    def forward(self, x):
        output = self.net(x)
        return output


class VGG(nn.Module):
    def __init__(self, classification=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 36
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 144
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 576
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 2304
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 73728
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1),  # Para: 409600
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=20),  # Para: 2560
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=classification),  # Para: 40
            nn.ReLU()
        )  # Total para: 1,309,212

    def forward(self, x):
        output = self.net(x)
        return output


class Joint_256(nn.Module):
    def __init__(self, classification=2):
        super().__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),  # Para: 400
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1),  # Para: 25600
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 26,000

        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 36
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 144
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 576
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 2304
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 73728
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # Para: 147456
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 897,012

        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1),  # Para: 1638400
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=64),  # Para: 16384
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=classification),  # Para: 256
            nn.ReLU()
        )  # Para: 1,655,040

    def forward(self, x_normal_160, x_sobel_32, x_canny_32):
        normal_vgg = self.vgg(x_normal_160)
        sobel_lenet = self.lenet(x_sobel_32)
        canny_lenet = self.lenet(x_canny_32)
        middle = torch.cat((sobel_lenet, normal_vgg, canny_lenet), 1)
        output = self.linear(middle)

        return output


class Joint_128(nn.Module):
    def __init__(self, classification=2):
        super().__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),  # Para: 150
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=1),  # Para: 4800
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 4950

        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 36
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 144
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 576
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 2304
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 4608
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 18432
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # Para: 36864
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 228,852

        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),  # Para: 147456
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),  # Para: 147456
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=32),  # Para: 4096
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=classification),  # Para: 128
            nn.ReLU()
        )  # Para: 299,136

    def forward(self, x_normal_160, x_sobel_32, x_canny_32):
        normal_vgg = self.vgg(x_normal_160)
        sobel_lenet = self.lenet(x_sobel_32)
        canny_lenet = self.lenet(x_canny_32)
        middle = torch.cat((sobel_lenet, normal_vgg, canny_lenet), 1)
        output = self.linear(middle)

        return output


class Joint_64(nn.Module):
    def __init__(self, classification=2):
        super().__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),  # Para: 150
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # Para: 2400
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 2550

        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 36
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),  # Para: 144
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),  # Para: 288
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),  # Para: 576
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 1152
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 2304
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # Para: 2304
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 4608
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Para: 9216
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Para: 57,492

        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # Para: 36864
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # Para: 36864
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=16),  # Para: 1024
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=classification),  # Para: 64
            nn.ReLU()
        )  # Para: 74,816

    def forward(self, x_normal_160, x_sobel_32, x_canny_32):
        normal_vgg = self.vgg(x_normal_160)
        sobel_lenet = self.lenet(x_sobel_32)
        canny_lenet = self.lenet(x_canny_32)
        middle = torch.cat((sobel_lenet, normal_vgg, canny_lenet), 1)
        output = self.linear(middle)

        return output