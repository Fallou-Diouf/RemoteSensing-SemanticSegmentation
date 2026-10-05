import torch
import torch.nn as nn


class Hypercolumns(nn.Module):
    def __init__(self):
        super(Hypercolumns, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)

        self.pool = nn.MaxPool2d(3)
        self.activation = nn.Tanh()

        self.interpol = nn.Upsample(
            size=(512, 512),
            mode="bilinear",
            align_corners=True
        )

        self.MLP1 = nn.Conv2d(256 + 128 + 64 + 64 + 3, 128, 1)
        self.MLP2 = nn.Conv2d(128, 128, 1)
        self.MLP3 = nn.Conv2d(128, 6, 1)

    def forward(self, x):

        x1 = self.activation(self.conv1(x))
        x1_p = self.pool(x1)

        x2 = self.activation(self.conv2(x1_p))
        x2_p = self.pool(x2)

        x3 = self.activation(self.conv3(x2_p))
        x3_p = self.pool(x3)

        x4 = self.activation(self.conv4(x3_p))

        x1_up = self.interpol(x1)
        x2_up = self.interpol(x2)
        x3_up = self.interpol(x3)
        x4_up = self.interpol(x4)

        hypercolumn = torch.cat( 
            (x, x1_up, x2_up, x3_up, x4_up),
            dim=1
        )

        x = self.activation(self.MLP1(hypercolumn))
        x = self.activation(self.MLP2(x))
        x = self.MLP3(x)

        return x