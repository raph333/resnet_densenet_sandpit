from torch import nn


class AllCNN(nn.Module):
    """
    All-convolutional network (no fully connected layers) with low number of
    parameters. Can be converted into a ResNet by 'residuals=True'
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
            nn.Dropout2d(p=0.2)
        )
        return layer

    def __init__(self, c=32, n_classes=100, residuals=False):
        """
        @ c: number of channels after first convolution
        @ n_classes: number of target classes
        """
        super().__init__()

        self.residuals = residuals
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = self.conv3x3(3,   c,   stride=1)
        self.conv2 = self.conv3x3(c,   c,   stride=1)

        self.conv3 = self.conv3x3(c,   c*2, stride=2)  # -> 16**2
        self.conv4 = self.conv3x3(c*2, c*2, stride=1)
        self.conv5 = self.conv3x3(c*2, c*2, stride=1)

        self.conv6 = self.conv3x3(c*2, c*4, stride=2)  # -> 8**2
        self.conv7 = self.conv3x3(c*4, c*4, stride=1)

        self.conv8 = nn.Sequential(
            nn.Conv2d(c*4, c*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c*4),
            nn.Dropout2d(p=0.2),
            self.relu
        )

        """
        conv9: make 1 channel per class -> take average of each feature map
        -> 100 real values: same as in fully connected output layer
        """
        self.conv9 = nn.Sequential(
            nn.Conv2d(c*4, n_classes, kernel_size=1, stride=1, padding=0),
            nn.Dropout2d(p=0.2),
            self.relu,
            nn.AvgPool2d(kernel_size=8)
        )

    def forward(self, x):

        res = int(self.residuals)  # True / False; if 0: residual becomes 0

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h) + h*res )

        h = self.relu(self.conv3(h))  # -> 16**2
        h = self.relu(self.conv4(h) + h*res )
        h = self.relu(self.conv5(h) + h*res )

        h = self.relu(self.conv6(h))  # -> 8**2
        h = self.relu(self.conv7(h) + h*res )

        h = self.relu(self.conv8(h) + h*res )
        y = self.conv9(h).squeeze()

        return y