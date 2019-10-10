from torch import nn


class ResNet(nn.Module):
    """
    Simple ResNet: residuals are only added to blocks of convolutional
    layers which keep the dimensions (height, width, channels) constant.
    """
  
    def conv_layer(self, in_channels, out_channels, stride=1):
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
  
  
    def conv_downsample(self, in_channels, out_channels):
        """
        Downsampling convolutional layers with stride 2
        The smart way to replace max-pooling: first, regular conv with stride 1,
        then conv layer with stride two instead of max-pooling
        see paper: "The all convolutional net"; downside: more parameters
        """
        layers = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )
        return layers
  
    def maxpool_downsample(self, in_channels, out_channels):
        layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        return layers
  
  
    def block(self, n_conv, in_channels, out_channels, stride=1):
        """
        Multiple convolutional blocks which can be skipped by 'shortcut'.
        First layer can change dimension, the other layers keep dimensions constant.
        Last block is without ReLU (ReLU after addition of residual)
        """
        first_conv = self.conv_layer(in_channels, out_channels, stride=stride)
        layers = list(first_conv.children())

        for _ in range(1, n_conv):
            block = self.conv_layer(out_channels, out_channels, stride=1)
            layers.extend([self.relu, *list(block.children())])

        return nn.Sequential(*layers)
  
  
    def __init__(self, residuals=True, block_size=2, c=32):
        """
        @ residuals: if False: convert into regluar conv-net
        @ block_size: number of layers to be skipped by a residual shortcut
        @ c: start with that many output channles in 1st conv
        """
        super().__init__()

        self.residuals = residuals
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = self.conv_layer(3,      c,    stride=1)
        self.block1 = self.block(block_size, c, c, stride=1)

        self.conv2 = self.conv_downsample(c, c*2)
        self.block2 = self.block(block_size, c*2, c*2, stride=1)

        self.conv3 = self.conv_downsample(c*2, c*4)
        self.block3 = self.block(block_size, c*4, c*4, stride=1)

        self.conv4 = self.conv_downsample(c*4, c*8)
        self.block4 = self.block(block_size, c*8, c*8, stride=1)

        self.fully_conntected = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(c*8 * 4**2, 1000, bias=True),
            self.relu,
            nn.Dropout(p=0.5),
            nn.Linear(1000, 100, bias=True)  # softmax calculated in CE-loss
        )
  
    
    def forward(self, x):

        res = int(self.residuals)  # True or False; if 0: residual becomes 0

        h = self.conv1(x)
        h = self.relu( self.block1(h) + h*res )

        h = self.conv2(h)
        h = self.relu( self.block2(h) + h*res )

        h = self.conv3(h)
        h = self.relu( self.block3(h) + h*res )

        h = self.conv4(h)
        h = self.relu( self.block4(h) + h*res )


        batch_size = h.size(0)
        channels = h.size(1)
        width, height = h.size(2), h.size(3)
        #print('from convolutions: ', batch_size, channels, width, height)
        h = h.view(batch_size, channels * width * height)
        #print('into feed-forward: ', h.shape)

        y = self.fully_conntected(h)
        #print('output: ', y.shape)
        return y