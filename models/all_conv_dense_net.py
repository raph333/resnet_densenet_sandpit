import torch
from torch import nn


class AllConvDenseNet(nn.Module):
    """
    All-convolutional DenseNet (no fully connected layers) with low number of
    parameters. The net downsamples 3 times from 32x32 to 4x4 and consists of 
    three dense blocks of four layers each.
    """

    def conv3x3(self, in_channels, out_channels, stride=1):
        """
        Basic 3x3 convolutional layer plus batch normalization
        Note: no ReLU in layer (ReLU after addition of residual)
        """
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )
        return layer

    def __init__(self, g=48, n_classes=100):
        """
        @ g: growth rate: every conv-layer outputs g channels
        @ n_classes: number of target classes
        """
        super().__init__()

        self.conv1a = self.conv3x3(3,   g, stride=1)
        self.conv1b = self.conv3x3(g,   g, stride=1)
        self.conv1c = self.conv3x3(g*2, g, stride=1)
        self.conv1d = self.conv3x3(g*3, g, stride=1)

        self.conv2a = self.conv3x3(g*4, g, stride=2)  # -> 16**2
        self.conv2b = self.conv3x3(g,   g, stride=1)
        self.conv2c = self.conv3x3(g*2, g, stride=1)
        self.conv2d = self.conv3x3(g*3, g, stride=1)

        self.conv3a = self.conv3x3(g*4, g, stride=2)  # -> 8**2
        self.conv3b = self.conv3x3(g,   g, stride=1)
        self.conv3c = self.conv3x3(g*2, g, stride=1)
        self.conv3d = self.conv3x3(g*3, g, stride=1)

        self.conv4 = nn.Sequential(
            nn.Conv2d(g*4, g*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(g*4),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )

        """
        conv9: make 1 channel per class -> take average of each feature map
        -> 100 real values: same as in fully connected output layer
        """
        self.conv_final = nn.Sequential(
            nn.Conv2d(g*4, n_classes, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )

    def forward(self, x):

        # dense block 1:
        h = self.conv1a(x)
        state = h
        #print(state.shape)
        h = self.conv1b(state)
        state = torch.cat( (state, h), dim=1 )  # add feature maps to 'memory'
        #print(state.shape)
        h = self.conv1c(state)
        state = torch.cat( (state, h), dim=1 )
        #print(state.shape)
        h = self.conv1d(state)
        state = torch.cat( (state, h), dim=1 )
        #print(state.shape)

        # dense block 2:
        h = self.conv2a(state)  # -> 16**2
        state = h
        h = self.conv2b(state)
        state = torch.cat( (state, h), dim=1 )
        h = self.conv2c(state)
        state = torch.cat( (state, h), dim=1 )
        h = self.conv2d(state)
        state = torch.cat( (state, h), dim=1 )

        # dense block 3:
        h = self.conv3a(state)  # -> 8**2
        state = h
        h = self.conv3b(state)
        state = torch.cat( (state, h), dim=1 )
        h = self.conv3c(state)
        state = torch.cat( (state, h), dim=1 )
        h = self.conv3d(state)
        state = torch.cat( (state, h), dim=1 )

        h = self.conv4(state)

        y = self.conv_final(h).squeeze()  # 192 channels -> 100x1 vector

        return y

