import torch
import torch.nn as nn
import random

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(36, 128, 2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, 2, padding=0)
        self.relu3 = nn.ReLU()
        #self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        #self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(3200, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64*64)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, boards):
        out = self.conv1(boards)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        #out = self.conv4(out)
        #out = out.view(out.shape[0], -1)
        #out = self.fc1(out)
        #out = self.relu4(out)
        #out = self.fc2(out)
        #out = self.softmax(out)
        #out = out.view(out.shape[0], 2, 8, 8)
        return out

model = TestModel()
i = 0
while True:
    print(i)
    x = torch.randn(random.randint(1000, 2000), 36, 8, 8)
    model(x)
    i += 1
