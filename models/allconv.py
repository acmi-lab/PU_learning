import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class AllConv(nn.Module):
    def __init__(self, inp_dim=(32, 32, 3), out_dim=2, hid_dim_full=32, bn=True):
        super(AllConv, self).__init__()
        self.bn = bn

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(32, 32, 1)
        self.conv6 = nn.Conv2d(32, 4, 1)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(16)
            self.bn3 = nn.BatchNorm2d(32)
            self.bn4 = nn.BatchNorm2d(32)
            self.bn5 = nn.BatchNorm2d(32)
            self.bn6 = nn.BatchNorm2d(4)

        self.conv_to_fc = 8*8*4
        self.fc1 = nn.Linear(self.conv_to_fc, hid_dim_full)
        self.fc2 = nn.Linear(hid_dim_full, int(hid_dim_full // 2))
        self.out = nn.Linear(int(hid_dim_full // 2), out_dim)

    def forward(self, x):
        if self.bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = F.relu(self.conv6(x))

        x = x.view(-1, self.conv_to_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # p = F.sigmoid(x)
        x = self.out(x)

        return x
