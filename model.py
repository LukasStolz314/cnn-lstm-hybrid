import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import corpus

class Conv3DModel(nn.Module):
    def __init__(self, num_classes=len(corpus)):
        super(Conv3DModel, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(5, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(5, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)

        # self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        # self.bn5 = nn.BatchNorm3d(1024)

        self.conv6 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn6= nn.BatchNorm3d(256)

        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # LSTM
        self.lstm_hidden_size = 256
        self.lstm_layers = 1
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)))
        # x = F.leaky_relu(self.bn5(self.conv5(x)))

        x = self.global_pool(x)

        x = F.leaky_relu(self.bn6(self.conv6(x)))

        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.3)
        return self.fc3(x)

