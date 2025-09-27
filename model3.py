dropout_value = 0.025 # Slightly decreased dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )
        # TRANSITION BLOCK 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        # TRANSITION BLOCK 2
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 1, padding=0)
        ) # output_size = 24
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),

            nn.Dropout(dropout_value)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(10),

            nn.Dropout(dropout_value)
        )


        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(10, 10, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.gap(x)
        x = self.conv10(x)
        x = x.view(-1, 10)
        # return F.log_softmax(x, dim=-1)
        return x