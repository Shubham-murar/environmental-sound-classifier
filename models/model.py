import torch.nn as nn
import torch.nn.functional as F

class SoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(SoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 32, 56, 56)
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
